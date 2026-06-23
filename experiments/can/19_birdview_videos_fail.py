"""Vidéos d'ÉCHECS du modèle 2 cams agentview + BIRDVIEW (16_proprio_birdview, 74.8 %).

Pendant de 12_wrist_sep_videos_fail.py mais pour le 16 (birdview).
Iter les val épisodes et garde les N_FAILS premiers échecs.
Vidéos en 2 vues côte à côte (agentview | birdview en 256px → 512×256/frame).

Sortie : results/runs/can/16_proprio_birdview/videos/birdview_FAIL_ep<idx>.mp4
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
OUT = Path("results/runs/can/16_proprio_birdview/videos")
N_FAILS = 6
MAX_TRIES = 40      # ~20 % de fail @500 -> ~6 fails dans les 25-35 premières val
HD = 256
POLICY_RES = 96
STEPS = 10
MAX_STEPS = 300


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def render(env, hd, cam):
    return env.env.sim.render(height=hd, width=hd, camera_name=cam)[::-1]


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
        obs = env.reset_to(dict(states=init_states[ep])); policy.reset()
        frames, success = [], False
        for _ in range(MAX_STEPS):
            ag_hd = render(env, HD, "agentview")
            bv_hd = render(env, HD, "birdview")
            frames.append(np.concatenate([ag_hd, bv_hd], axis=1))
            ag_p = render(env, POLICY_RES, "agentview")
            bv_p = render(env, POLICY_RES, "birdview")
            ag_t = torch.from_numpy(ag_p.copy()).permute(2, 0, 1).float() / 255.0
            bv_t = torch.from_numpy(bv_p.copy()).permute(2, 0, 1).float() / 255.0
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
                success = True; break
        tag = "OK" if success else "FAIL"
        print(f"  ep {val_idx[ep]} -> {tag} ({len(frames)} frames)", flush=True)
        if not success:
            out = OUT / f"birdview_FAIL_ep{val_idx[ep]}.mp4"
            imageio.mimsave(out, frames, fps=20)
            print(f"    -> sauvé {out}", flush=True)
            fails += 1
            if fails >= N_FAILS:
                break
    print(f"\n{fails} échec(s) filmé(s) dans {OUT}")


if __name__ == "__main__":
    main()
