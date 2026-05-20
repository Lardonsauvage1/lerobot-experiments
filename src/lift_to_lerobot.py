"""Conversion du dataset Robomimic Lift PH au format LeRobot.

Robomimic Lift PH = HDF5 avec 200 démos humaines.
LeRobot = parquet + vidéos + metadata (format natif HuggingFace).

On replay chaque démo dans le sim Robosuite pour rendre les images,
puis on les écrit au format LeRobot via LeRobotDataset.create().

Le dataset résultant est stocké localement (data_cache/lerobot_lift_ph/)
et utilisable avec lerobot-train via --dataset.root=...
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import numpy as np
import h5py
import torch
from pathlib import Path
from src.lift_data import STATE_KEYS

# ============================================================
HDF5_PATH = "data_cache/robomimic_lift_ph/low_dim_v15.hdf5"
OUTPUT_ROOT = Path("data_cache/lerobot_lift_ph")
REPO_ID = "local/lift_ph"
IMAGE_SIZE = 96       # même que nos runs précédents
CAMERA_NAME = "agentview"
FPS = 20              # control_freq Robosuite
TASK_DESCRIPTION = "Pick up the cube and lift it."
# ============================================================


def convert():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs

    # Cleanup si existe déjà
    if OUTPUT_ROOT.exists():
        print(f"⚠️  {OUTPUT_ROOT} existe déjà. Supprime-le manuellement si tu veux relancer.")
        return

    # Définir les features LeRobot
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (19,),  # eef_pos(3) + eef_quat(4) + gripper_qpos(2) + object(10)
            "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                      "eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z",
                      "gripper_l", "gripper_r",
                      "obj0", "obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7", "obj8", "obj9"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        },
    }

    print(f"Création du dataset LeRobot local : {OUTPUT_ROOT}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=OUTPUT_ROOT,
        robot_type="panda",
        use_videos=True,
        video_backend="pyav",  # compatible Mac
    )

    # Setup env pour render
    print("Setup env Robosuite Lift...")
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=CAMERA_NAME,
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=FPS, reward_shaping=False,
    )

    # Iterate over demos
    with h5py.File(HDF5_PATH, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"\n{len(demo_names)} démos à convertir...")

        for d_idx, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            states_demo = demo["states"][:]
            actions_demo = demo["actions"][:]
            obs_grp = demo["obs"]
            state_low = np.concatenate([obs_grp[k][:] for k in STATE_KEYS], axis=1)
            n = states_demo.shape[0]

            env.reset()
            for i in range(n):
                env.sim.set_state_from_flattened(states_demo[i])
                env.sim.forward()
                img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name=CAMERA_NAME)[::-1]
                # LeRobot attend (C, H, W) uint8
                img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))

                frame = {
                    "observation.image": img_chw,
                    "observation.state": state_low[i].astype(np.float32),
                    "action": actions_demo[i].astype(np.float32),
                    "task": TASK_DESCRIPTION,
                }
                dataset.add_frame(frame)

            dataset.save_episode()

            if (d_idx + 1) % 20 == 0:
                print(f"  {d_idx + 1}/{len(demo_names)} démos écrites")

    env.close()

    print("\nConsolidation du dataset...")
    # LeRobot v0.5 : consolidate happens automatically on save_episode in streaming mode
    # mais on peut forcer le flush des stats

    print(f"\n✓ Dataset LeRobot créé : {OUTPUT_ROOT}")
    print(f"  Frames total : {dataset.num_frames}")
    print(f"  Épisodes : {dataset.num_episodes}")
    print(f"\nPour entraîner :")
    print(f"  lerobot-train --policy.type=diffusion \\")
    print(f"    --dataset.repo_id={REPO_ID} \\")
    print(f"    --dataset.root={OUTPUT_ROOT}")


if __name__ == "__main__":
    convert()
