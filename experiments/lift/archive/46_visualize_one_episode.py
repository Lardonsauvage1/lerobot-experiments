"""Sauve une vidéo d'1 épisode + log de toutes les actions, pour debug visuel."""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import numpy as np
import torch
import imageio
from src.lift_data import build_state_vector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_step", type=str, default="012000",
                    help="step folder name: 003000 / 006000 / 009000 / 012000")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--save_video", action="store_true", default=False)
args = parser.parse_args()

CHECKPOINT = f"results/runs/lift/46_diffusion_official/checkpoints/{args.checkpoint_step}/pretrained_model"
OUT_VIDEO = f"results/runs/lift/46_diffusion_official/debug_episode_{args.checkpoint_step}.mp4"
IMAGE_SIZE = 96
MAX_STEPS = args.max_steps


def main():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    import robosuite as rs

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    policy = DiffusionPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    policy.diffusion.num_inference_steps = 10
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )

    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )

    import h5py
    with h5py.File("data_cache/robomimic_lift_ph/low_dim_v15.hdf5", "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        init_states = np.stack([f["data"][d]["states"][0] for d in demos])
    rng = np.random.default_rng(seed=42)
    demo_idx = int(rng.integers(0, init_states.shape[0]))
    env.reset()
    env.sim.set_state_from_flattened(init_states[demo_idx])
    env.sim.forward()
    obs = env._get_observations()
    print(f"Using demo init #{demo_idx}")
    policy.reset()
    cube_pos_init = obs.get("cube_pos")
    eef_pos_init = obs.get("robot0_eef_pos")
    print(f"Init cube_pos: {cube_pos_init}")
    print(f"Init eef_pos: {eef_pos_init}")
    print(f"Distance init: {np.linalg.norm(np.array(cube_pos_init) - np.array(eef_pos_init)):.3f}")

    frames = []
    actions_log = []
    for step in range(MAX_STEPS):
        frame_hd = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
        frames.append(frame_hd)

        img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        state_np = build_state_vector(obs).astype(np.float32)
        state_t = torch.tensor(state_np)

        obs_dict = {
            "observation.image": img_t.unsqueeze(0).to(device),
            "observation.state": state_t.unsqueeze(0).to(device),
        }
        obs_dict = preprocessor(obs_dict)
        with torch.no_grad():
            action_norm = policy.select_action(obs_dict)
        action = postprocessor(action_norm).squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        actions_log.append(action.copy())

        obs, _, _, _ = env.step(action)

        if step % 20 == 0:
            cube_z = obs.get("cube_pos", [0, 0, 0])[2]
            eef = obs.get("robot0_eef_pos", [0, 0, 0])
            print(f"  step {step:3d}: action=[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f} "
                  f"{action[3]:+.2f} {action[4]:+.2f} {action[5]:+.2f} {action[6]:+.2f}] "
                  f"| eef=[{eef[0]:+.2f} {eef[1]:+.2f} {eef[2]:+.2f}] | cube_z={cube_z:.3f}")

    env.close()

    actions_log = np.array(actions_log)
    print(f"\n=== Action stats sur 200 steps ===")
    for i, n in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "grip"]):
        a = actions_log[:, i]
        print(f"  {n:5s}: min={a.min():+.2f}  max={a.max():+.2f}  mean={a.mean():+.2f}  std={a.std():.3f}")

    # Save video
    if args.save_video:
        print(f"\nSaving video to {OUT_VIDEO}")
        imageio.mimsave(OUT_VIDEO, frames, fps=20)
    print("Done.")


if __name__ == "__main__":
    main()
