"""Debug : comparer la magnitude des actions sorties par le policy vs celles du dataset.

Si policy actions ≪ dataset actions → mismatch preprocessing/normalization.
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import numpy as np
import torch
import h5py
from pathlib import Path
from src.lift_data import build_state_vector, STATE_KEYS

CHECKPOINT = "results/runs/lift/46_diffusion_official/checkpoints/012000/pretrained_model"
HDF5_PATH = "data_cache/robomimic_lift_ph/low_dim_v15.hdf5"
IMAGE_SIZE = 96
N_SAMPLES = 20


def stats(name, arr):
    arr = np.asarray(arr)
    print(f"  {name:25s} shape={arr.shape}  "
          f"min={arr.min():+.3f}  max={arr.max():+.3f}  "
          f"mean={arr.mean():+.3f}  std={arr.std():.3f}  "
          f"|mean|abs={np.abs(arr).mean():.3f}")


def main():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    import robosuite as rs

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"=== Loading policy + processors ===")
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
    print(f"  Loaded.")

    # ===== Actions du dataset (référence) =====
    print(f"\n=== Dataset actions (référence — ce que le policy devrait sortir) ===")
    with h5py.File(HDF5_PATH, "r") as f:
        # Concatène toutes les actions des 200 démos
        all_actions = np.concatenate([f["data"][k]["actions"][:] for k in f["data"].keys()], axis=0)
    print(f"  Total frames dataset : {all_actions.shape[0]}")
    for i, dim in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "grip"]):
        stats(f"action[{dim}]", all_actions[:, i])

    # ===== Setup env, get observations =====
    print(f"\n=== Setup env Robosuite ===")
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )

    # ===== Actions du policy =====
    print(f"\n=== Policy actions (sur {N_SAMPLES} obs simulées) ===")
    policy_actions = []
    obs = env.reset()
    policy.reset()
    for i in range(N_SAMPLES):
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
        policy_actions.append(action)
        obs, _, _, _ = env.step(np.clip(action, -1, 1))
    policy_actions = np.array(policy_actions)

    for i, dim in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "grip"]):
        stats(f"action[{dim}]", policy_actions[:, i])

    # ===== Image et state actuels =====
    print(f"\n=== Format observation ===")
    print(f"  Image (img_t): dtype={img_t.dtype}  shape={img_t.shape}  "
          f"min={img_t.min():.3f}  max={img_t.max():.3f}  mean={img_t.mean():.3f}")
    print(f"  State (state_np): shape={state_np.shape}  min={state_np.min():+.3f}  max={state_np.max():+.3f}")

    # ===== Comparison =====
    print(f"\n=== COMPARAISON ===")
    print(f"  Ratio |policy_mean_abs| / |dataset_mean_abs| par dim :")
    for i, dim in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "grip"]):
        ratio = np.abs(policy_actions[:, i]).mean() / max(1e-6, np.abs(all_actions[:, i]).mean())
        flag = " ← TROP PETIT" if ratio < 0.3 else (" ← TROP GROS" if ratio > 3 else "")
        print(f"    {dim:5s} : {ratio:6.3f}{flag}")

    print(f"\n=== Preprocessor info ===")
    if hasattr(policy, "policy_preprocessor"):
        print(f"  Has policy_preprocessor : {type(policy.policy_preprocessor).__name__}")
    if hasattr(policy, "policy_postprocessor"):
        print(f"  Has policy_postprocessor : {type(policy.policy_postprocessor).__name__}")

    env.close()


if __name__ == "__main__":
    main()
