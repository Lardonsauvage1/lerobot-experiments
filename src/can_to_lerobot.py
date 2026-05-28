"""Conversion du dataset Robomimic Can (PickPlaceCan) PH au format LeRobot.

Même pipeline que lift_to_lerobot.py (replay des états dans le sim → rendu agentview
→ écriture LeRobot), adapté à Can : env PickPlaceCan, object 14D (state total 23D),
horizon ~2× plus long que Lift.

  python -u src/can_to_lerobot.py            # conversion complète (200 démos)
  python -u src/can_to_lerobot.py --smoke    # 2 démos, vérif rendu + dataset.create
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path

import h5py
import numpy as np

HDF5_PATH = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUTPUT_ROOT = Path("data_cache/lerobot_can_ph")
REPO_ID = "local/can_ph"
IMAGE_SIZE = 96
CAMERA_NAME = "agentview"
FPS = 20
TASK_DESCRIPTION = "Pick up the can and place it in the bin."
# State Can = proprio + position canette = 12D (voir src/can_eval.py pour le pourquoi).
# ⚠️ Côté DATASET (1.4), la can_pos est dans object[0:3] (côté env live 1.5 : object[7:10]).
STATE_DIM = 3 + 4 + 2 + 3  # eef_pos + eef_quat + gripper_qpos + can_pos(3) = 12


def convert(proprio=False, smoke=False):
    """proprio=True : state 9D = proprio SEULE (eef_pos+eef_quat+gripper), SANS can_pos.
    Test de transférabilité réel sur Can : le modèle doit VOIR la canette (image) au lieu
    qu'on lui souffle sa position. Sort dans lerobot_can_ph_proprio / local/can_ph_proprio."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import robosuite as rs

    suffix = "_proprio" if proprio else ""
    out_root = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + suffix)
    repo_id = REPO_ID + suffix
    out = out_root.with_name(out_root.name + "_smoke") if smoke else out_root
    if out.exists():
        print(f"⚠️  {out} existe déjà. Supprime-le pour relancer.")
        return

    base_names = ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_w", "eef_quat_x",
                  "eef_quat_y", "eef_quat_z", "gripper_l", "gripper_r"]
    if proprio:
        state_dim, state_names = 9, base_names                          # SANS can_pos
    else:
        state_dim, state_names = 12, base_names + ["can_x", "can_y", "can_z"]
    features = {
        "observation.image": {"dtype": "video", "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
                              "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
    }

    print(f"Création dataset LeRobot : {out} (proprio={proprio}, state={state_dim}D)")
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=out,
                                    robot_type="panda", use_videos=True, video_backend="pyav")

    print("Setup env Robosuite PickPlaceCan...")
    env = rs.make(env_name="PickPlaceCan", robots="Panda",
                  has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=False, camera_names=CAMERA_NAME,
                  camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
                  control_freq=FPS, reward_shaping=False)

    with h5py.File(HDF5_PATH, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if smoke:
            demo_names = demo_names[:2]
        print(f"\n{len(demo_names)} démos à convertir...")
        for d_idx, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            states_demo = demo["states"][:]
            actions_demo = demo["actions"][:]
            o = demo["obs"]
            if proprio:
                state_low = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                                            o["robot0_gripper_qpos"][:]], axis=1)  # 9D
            else:
                # 12D = eef_pos(3) + eef_quat(4) + gripper(2) + can_pos = object[0:3] (dataset 1.4)
                state_low = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                                            o["robot0_gripper_qpos"][:], o["object"][:, 0:3]], axis=1)
            assert state_low.shape[1] == state_dim, f"state dim {state_low.shape[1]} != {state_dim}"
            env.reset()
            for i in range(states_demo.shape[0]):
                env.sim.set_state_from_flattened(states_demo[i])
                env.sim.forward()
                img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name=CAMERA_NAME)[::-1]
                dataset.add_frame({
                    "observation.image": np.ascontiguousarray(img.transpose(2, 0, 1)),
                    "observation.state": state_low[i].astype(np.float32),
                    "action": actions_demo[i].astype(np.float32),
                    "task": TASK_DESCRIPTION,
                })
            dataset.save_episode()
            if (d_idx + 1) % 20 == 0:
                print(f"  {d_idx + 1}/{len(demo_names)} démos", flush=True)

    env.close()
    print(f"\n✓ Dataset créé : {out} | frames={dataset.num_frames} épisodes={dataset.num_episodes}")
    if smoke:
        # vérif : sauver une image pour inspection visuelle
        import imageio
        f = h5py.File(HDF5_PATH, "r")
        d0 = f["data"]["demo_0"]
        env2 = rs.make(env_name="PickPlaceCan", robots="Panda", has_renderer=False,
                       has_offscreen_renderer=True, use_camera_obs=False, camera_names=CAMERA_NAME,
                       camera_heights=256, camera_widths=256, control_freq=FPS)
        env2.reset(); env2.sim.set_state_from_flattened(d0["states"][len(d0["states"])//2]); env2.sim.forward()
        mid = env2.sim.render(height=256, width=256, camera_name=CAMERA_NAME)[::-1]
        imageio.imwrite("data_cache/can_render_check.png", mid)
        print("  rendu mi-démo sauvé : data_cache/can_render_check.png (vérifie que c'est bien la scène Can)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proprio", action="store_true", help="state 9D proprio seule (sans can_pos)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    convert(proprio=a.proprio, smoke=a.smoke)
