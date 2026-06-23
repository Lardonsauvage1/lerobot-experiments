"""Vidéo d'un rollout du modèle 16 (no-aug, 80%) à la résolution NATIVE 96x96.
C'est exactement ce que voit le réseau, sans aucun upscale ni augmentation.

Utilité : comprendre visuellement la quantité d'info que la policy perçoit
réellement (à comparer aux vidéos 256x256 trompeuses).

Cible : ep152 (échec de référence — on a déjà la version 256).
Sortie : results/runs/can/16_proprio_birdview/videos/native96_ep152.mp4
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

CKPT = "results/runs/can/16_proprio_birdview/checkpoints/010000/pretrained_model"
OUT = Path("results/runs/can/16_proprio_birdview/videos/native96_ep152.mp4")
TARGET_VAL_IDX = 2          # ep 152 dans la val (split[2])
RES = 96                    # NATIVE — ce que voit le réseau
STEPS = 10
MAX_STEPS = 300


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    val_idx = can_eval.load_or_make_split()["val"]
    ep_id = val_idx[TARGET_VAL_IDX]
    init_states = can_eval.load_init_states([ep_id])
    print(f"Cible : val[{TARGET_VAL_IDX}] = ep {ep_id}", flush=True)

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

    obs = env.reset_to(dict(states=init_states[0])); policy.reset()
    frames = []
    for _ in range(MAX_STEPS):
        ag = env.env.sim.render(height=RES, width=RES, camera_name="agentview")[::-1]
        bv = env.env.sim.render(height=RES, width=RES, camera_name="birdview")[::-1]
        frames.append(np.concatenate([ag, bv], axis=1))
        ag_t = torch.from_numpy(ag.copy()).permute(2, 0, 1).float() / 255.0
        bv_t = torch.from_numpy(bv.copy()).permute(2, 0, 1).float() / 255.0
        state_t = torch.from_numpy(state_proprio(obs))
        obs_dict = pre({
            "observation.images.agentview": ag_t.unsqueeze(0).to(device),
            "observation.images.birdview": bv_t.unsqueeze(0).to(device),
            "observation.state": state_t.unsqueeze(0).to(device),
        })
        with torch.no_grad():
            a = policy.select_action(obs_dict)
        a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
        obs = env.step(a)[0]
        if env.is_success()["task"]:
            break
    imageio.mimsave(OUT, frames, fps=20)
    print(f"Sauvé : {OUT} | {len(frames)} frames | shape={frames[0].shape}", flush=True)


if __name__ == "__main__":
    main()
