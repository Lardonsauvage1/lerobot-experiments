"""Vidéos des ÉCHECS du modèle Can vision-only — itère les val épisodes et garde N_FAILS échecs.

Pendant de 04_proprio_videos.py (qui montrait des succès). Ici on cherche à voir COMMENT
ça échoue : la canette pas saisie ? Saisie mais lâchée ? Mauvais bac ? Bout des 300 steps ?

Sortie : results/runs/can/02_proprio/videos/proprio_FAIL_ep<idx>.mp4
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

from pathlib import Path

import imageio
import numpy as np
import torch

from src import can_eval

CKPT = "results/runs/can/02_proprio/checkpoints/010000/pretrained_model"
OUT = Path("results/runs/can/02_proprio/videos")
N_FAILS = 3
MAX_TRIES = 25       # garde-fou (à ~28% de fail, ~3 fails attendus dans les 11 premiers)
HD = 256
STEPS = 10
MAX_STEPS = 300      # comme dans l'éval (Can plus long)


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    val_idx = can_eval.load_or_make_split()["val"][:MAX_TRIES]
    init_states = can_eval.load_init_states(val_idx)
    env = can_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    policy = DiffusionPolicy.from_pretrained(CKPT).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    fails = 0
    for ep in range(len(init_states)):
        obs = env.reset_to(dict(states=init_states[ep]))
        policy.reset()
        frames, success = [], False
        for _ in range(MAX_STEPS):
            frames.append(env.env.sim.render(height=HD, width=HD, camera_name="agentview")[::-1])
            img = env.env.sim.render(height=96, width=96, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            state_t = torch.from_numpy(state_proprio(obs))
            obs_dict = pre({"observation.image": img_t.unsqueeze(0).to(device),
                            "observation.state": state_t.unsqueeze(0).to(device)})
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs = env.step(a)[0]
            if env.is_success()["task"]:
                success = True; break
        tag = "OK" if success else "FAIL"
        print(f"  ep {val_idx[ep]} -> {tag} ({len(frames)} frames)", flush=True)
        if not success:
            out = OUT / f"proprio_FAIL_ep{val_idx[ep]}.mp4"
            imageio.mimsave(out, frames, fps=20)
            print(f"    -> sauvé {out}", flush=True)
            fails += 1
            if fails >= N_FAILS:
                break
    print(f"\n{fails} échec(s) filmé(s) dans {OUT}")


if __name__ == "__main__":
    main()
