"""Vidéos du modèle Can vision-only (9D proprio, SANS coords canette) — best ckpt 02_proprio.

Filme N_EP rollouts val en 256px (env PickPlaceCan, état 9D, max 250 steps).
Sortie : results/runs/can/02_proprio/videos/proprio_ep<idx>_<OK|FAIL>.mp4
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
N_EP = 3
HD = 256
STEPS = 10
MAX_STEPS = 250


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    val_idx = can_eval.load_or_make_split()["val"][:N_EP]
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

    for ep in range(N_EP):
        obs = env.reset_to(dict(states=init_states[ep]))
        policy.reset()
        frames, success, hold = [], False, 0
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
                success = True; hold += 1
                if hold > 10:
                    break
        tag = "OK" if success else "FAIL"
        out = OUT / f"proprio_ep{val_idx[ep]}_{tag}.mp4"
        imageio.mimsave(out, frames, fps=20)
        print(f"  ep {val_idx[ep]} -> {tag} ({len(frames)} frames) {out}", flush=True)
    print(f"\nVidéos : {OUT}")


if __name__ == "__main__":
    main()
