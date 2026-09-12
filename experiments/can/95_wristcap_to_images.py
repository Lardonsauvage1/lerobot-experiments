#!/usr/bin/env python
"""Convertit un dataset LeRobot VIDEO -> IMAGES (use_videos=False) pour gb10 (torchcodec HS sur aarch64).
Générique : préserve toutes les features (images en PNG, state/action inchangés), metadata_buffer_size=1.
Usage : python 95_wristcap_to_images.py SRC_REPO SRC_ROOT DST_REPO DST_ROOT
Smoke-test : LIMIT=60 python 95_...py ...  (ne traite que 60 frames, écrit puis s'arrête)."""
import os, shutil, sys
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC_REPO, SRC_ROOT, DST_REPO, DST_ROOT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
LIMIT = int(os.environ.get("LIMIT", "0"))          # 0 = tout
SKIP = {"index", "episode_index", "frame_index", "timestamp", "task_index",
        "next.done", "next.reward", "next.success", "task"}

src = LeRobotDataset(SRC_REPO, root=SRC_ROOT)
N = src.num_frames if LIMIT <= 0 else min(LIMIT, src.num_frames)
print(f"source : {src.meta.total_episodes} ép, {src.num_frames} frames (on traite {N})")
if DST_ROOT.exists():
    shutil.rmtree(DST_ROOT)

feats, img_keys = {}, []
for k, v in src.meta.features.items():
    if k in SKIP:
        continue
    if v["dtype"] in ("video", "image"):
        feats[k] = {"dtype": "image", "shape": tuple(v["shape"]), "names": v.get("names")}
        img_keys.append(k)
    else:
        feats[k] = {"dtype": v["dtype"], "shape": tuple(v["shape"]), "names": v.get("names")}
print("features:", {k: feats[k]["dtype"] for k in feats}, "| images:", img_keys)

dst = LeRobotDataset.create(repo_id=DST_REPO, fps=int(src.fps), features=feats, root=DST_ROOT,
                            use_videos=False, image_writer_processes=0, image_writer_threads=4,
                            metadata_buffer_size=1)   # flush méta par ép (sinon derniers ép perdus)
# tâche : reprend celle de la source si dispo, sinon libellé générique
TASK = src[0].get("task") if isinstance(src[0].get("task"), str) else "pick and place the can"

def to_hwc_u8(t):
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):        # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:                          # float [0,1] -> uint8
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)

cur_ep = 0
for i in range(N):
    f = src[i]
    ep = int(f["episode_index"])
    if ep != cur_ep:
        dst.save_episode()
        print(f"  ép {cur_ep} sauvé (cumul {i})", flush=True)
        cur_ep = ep
    frame = {}
    for k in feats:
        frame[k] = to_hwc_u8(f[k]) if k in img_keys else (f[k].numpy() if hasattr(f[k], "numpy") else f[k])
    frame["task"] = TASK
    dst.add_frame(frame)
dst.save_episode()
print(f"TERMINE : {dst.meta.total_episodes} ép -> {DST_ROOT}")
