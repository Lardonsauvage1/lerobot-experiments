"""Phase 4 — vidéos MP4 d'épisodes pour comparer visuellement les modèles du sweep.

Pour chaque modèle (baseline + 3 tailles U-Net), rejoue les MÊMES N épisodes du val set
(mêmes init states → comparaison à scènes identiques), rend en HD (256×256) et sauve un MP4
par épisode. Utilise le harnais validé (env Robomimic + sign fix), pas rs.make().

Sortie : results/runs/lift/51_unet_sweep_eval/videos/<label>_ep<idx>_<OK|FAIL>.mp4
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from src import lift_eval

OUT = Path("results/runs/lift/51_unet_sweep_eval/videos")
HD = 256
NUM_INFERENCE_STEPS = 10
MAX_STEPS = 200

# (label, checkpoint pretrained_model) — meilleurs checkpoints (val min) du sweep + baseline
MODELS = [
    ("baseline_252M", "results/runs/lift/47_baseline_150/checkpoints/006000/pretrained_model"),
    ("unet_256_512_1024_76M", "results/runs/lift/51_unet_d256_512_1024/checkpoints/004500/pretrained_model"),
    ("unet_128_256_512_29M", "results/runs/lift/51_unet_d128_256_512/checkpoints/004500/pretrained_model"),
    ("unet_64_128_256_16M", "results/runs/lift/51_unet_d64_128_256/checkpoints/006000/pretrained_model"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3, help="nb d'épisodes val à filmer (mêmes pour tous)")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    for label, ckpt in MODELS:
        print(f"\n##### {label} ({ckpt}) #####", flush=True)
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

        for ep in range(len(val_idx)):
            obs = lift_eval.fix_obs_sign(env.reset_to(dict(states=init_states[ep])))
            policy.reset()
            frames, success = [], False
            for _ in range(MAX_STEPS):
                frames.append(env.env.sim.render(height=HD, width=HD, camera_name="agentview")[::-1])
                img = env.env.sim.render(height=96, width=96, camera_name="agentview")[::-1]
                img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                state_t = torch.from_numpy(lift_eval.build_state_vector(obs))
                obs_dict = pre({"observation.image": img_t.unsqueeze(0).to(device),
                                "observation.state": state_t.unsqueeze(0).to(device)})
                with torch.no_grad():
                    a = policy.select_action(obs_dict)
                a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
                obs = lift_eval.fix_obs_sign(env.step(a)[0])
                if env.is_success()["task"]:
                    success = True
            tag = "OK" if success else "FAIL"
            out = OUT / f"{label}_ep{val_idx[ep]}_{tag}.mp4"
            imageio.mimsave(out, frames, fps=20)
            print(f"  ep {val_idx[ep]} -> {tag}  ({out})", flush=True)
        del policy, pre, post

    print(f"\nVidéos dans : {OUT}")


if __name__ == "__main__":
    main()
