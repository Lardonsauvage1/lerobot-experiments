"""Phase 5 — vidéos N=10 : comparer 4 pas vs 10 pas de diffusion sur les mêmes scènes.

Le modèle N=10 (mini-CNN) passe de 84 % @4 pas à 94 % @10 pas. On filme les MÊMES 3 épisodes
val aux deux réglages pour voir la différence de comportement.

Sortie : results/runs/lift/63_dataeff_n10/videos/n10_<steps>pas_ep<idx>_<OK|FAIL>.mp4
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

from pathlib import Path

import imageio
import numpy as np
import torch

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

CKPT = "results/runs/lift/63_dataeff_n10/checkpoints/003000/pretrained_model"
OUT = Path("results/runs/lift/63_dataeff_n10/videos")
HD = 256
N_EP = 3
STEPS_LIST = [4, 10]
MAX_STEPS = 200


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS_LIST)
    steps_list = ap.parse_args().steps
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    val_idx = lift_eval.load_or_make_split()["val"][:N_EP]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    policy = load_minicnn_policy(CKPT, device)
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    for steps in steps_list:
        policy.diffusion.num_inference_steps = steps
        print(f"\n##### {steps} pas #####", flush=True)
        for ep in range(N_EP):
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
            out = OUT / f"n10_{steps}pas_ep{val_idx[ep]}_{tag}.mp4"
            imageio.mimsave(out, frames, fps=20)
            print(f"  ep {val_idx[ep]} -> {tag}  ({out})", flush=True)

    print(f"\nVidéos dans : {OUT}")


if __name__ == "__main__":
    main()
