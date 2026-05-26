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


def convert(proprio=False):
    """proprio=True : state 9D = proprio SEULE (eef_pos+eef_quat+gripper), SANS les coords du cube.
    Test de transférabilité réel : le modèle doit VOIR le cube (image) au lieu qu'on lui souffle
    sa position. Sort dans lerobot_lift_ph_proprio / local/lift_ph_proprio."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs

    out_root = OUTPUT_ROOT.with_name("lerobot_lift_ph_proprio") if proprio else OUTPUT_ROOT
    repo_id = "local/lift_ph_proprio" if proprio else REPO_ID
    if out_root.exists():
        print(f"⚠️  {out_root} existe déjà. Supprime-le manuellement si tu veux relancer.")
        return

    base_names = ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_w", "eef_quat_x",
                  "eef_quat_y", "eef_quat_z", "gripper_l", "gripper_r"]
    if proprio:
        state_shape, state_names = (9,), base_names              # SANS object
    else:
        state_shape, state_names = (19,), base_names + [f"obj{i}" for i in range(10)]
    features = {
        "observation.image": {"dtype": "video", "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
                              "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": state_shape, "names": state_names},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
    }

    print(f"Création du dataset LeRobot local : {out_root} (proprio={proprio}, state={state_shape[0]}D)")
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=out_root,
                                    robot_type="panda", use_videos=True, video_backend="pyav")

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
            if proprio:
                state_low = np.concatenate([obs_grp["robot0_eef_pos"][:], obs_grp["robot0_eef_quat"][:],
                                            obs_grp["robot0_gripper_qpos"][:]], axis=1)  # 9D, sans object
            else:
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
    print(f"\n✓ Dataset LeRobot créé : {out_root} | frames={dataset.num_frames} épisodes={dataset.num_episodes}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proprio", action="store_true", help="state 9D proprio seule (sans coords cube)")
    convert(proprio=ap.parse_args().proprio)
