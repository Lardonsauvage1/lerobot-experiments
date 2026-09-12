#!/usr/bin/env python
"""Reconstruit le dataset propre en IMAGES (use_videos=False) -> zéro décodage vidéo (robuste sur gb10).
Lit le dataset vidéo propre (décodage OK sur Mac) et réécrit chaque frame en PNG.
Sortie : data_cache/lerobot_apple_joint_224_clean_img."""
import shutil
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC_REPO, SRC_ROOT = "local/apple_joint_224_clean", "data_cache/lerobot_apple_joint_224_clean"
DST_REPO, DST_ROOT = "local/apple_joint_224_clean_img", Path("data_cache/lerobot_apple_joint_224_clean_img")
IMG = 224

src = LeRobotDataset(SRC_REPO, root=SRC_ROOT)          # tout, sans delta_timestamps
N = src.num_frames
print(f"source : {src.meta.total_episodes} ép, {N} frames")
if DST_ROOT.exists(): shutil.rmtree(DST_ROOT)

feats = {
  "observation.images.left":  {"dtype": "image", "shape": (3, IMG, IMG), "names": ["channels", "height", "width"]},
  "observation.images.right": {"dtype": "image", "shape": (3, IMG, IMG), "names": ["channels", "height", "width"]},
  "observation.state": {"dtype": "float32", "shape": (5,), "names": [f"joint_{i+1}" for i in range(5)]},
  "action": {"dtype": "float32", "shape": (6,), "names": [f"jt_{i+1}" for i in range(5)] + ["gripper"]},
}
dst = LeRobotDataset.create(repo_id=DST_REPO, fps=int(src.fps), features=feats, root=DST_ROOT,
                            robot_type="real5dof", use_videos=False, image_writer_processes=0,
                            image_writer_threads=4, metadata_buffer_size=1)   # flush méta à chaque ép (sinon 6 derniers perdus)
TASK = src[0]["task"] if isinstance(src[0].get("task"), str) else "pick the white apple and place it at the drop point"

def to_hwc_u8(t):
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

cur_ep, done = 0, 0
for i in range(N):
    f = src[i]
    ep = int(f["episode_index"])
    if ep != cur_ep:                                   # frontière d'épisode -> save
        dst.save_episode(); done += 1; cur_ep = ep
        print(f"  ép {done} sauvé (cumul {i} frames)", flush=True)
    dst.add_frame({
        "observation.images.left":  to_hwc_u8(f["observation.images.left"]),
        "observation.images.right": to_hwc_u8(f["observation.images.right"]),
        "observation.state": f["observation.state"].numpy(),
        "action": f["action"].numpy(),
        "task": TASK,
    })
dst.save_episode()
print(f"TERMINE: {dst.meta.total_episodes} ép, {N} frames (images) -> {DST_ROOT}")
