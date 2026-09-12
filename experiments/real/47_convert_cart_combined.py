#!/usr/bin/env python
"""Combine les 3 collections CARTÉSIENNES (_cart) en 1 dataset LeRobot — caméra FIXE, 128px, 15 Hz, EN IMAGES.
Ordre : classique (saisies) -> correction -> correction far. Action = [tcp(t+1)[6], gripper] = 7D.
Sortie : data_cache/lerobot_apple_cart_combined_128 (repo local/apple_cart_combined_128).
Affiche les plages d'index par source (pour caler train/val)."""
import shutil
from pathlib import Path
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FS = 15.0
RES = 128
TASK = "pick the white apple and place it at the drop point"
BASE = Path("data_cache/_cart_raw")
SRCS = [
    ("classique", BASE / "batch_collect_20260716_023632_cart"),
    ("correction", BASE / "batch_recovery_20260722_175404_cart"),
    ("correction_far", BASE / "batch_recovery_far_20260722_183001_cart"),
]
TS = get_typestore(Stores.ROS2_JAZZY)
FIXED, TCP, GRIP = "/head_camera/left/image_raw/compressed", "/tcp_pose", "/gripper"


def read_bag(bag_dir):
    pt, pv, ft, fraw, gt, gv = [], [], [], [], [], []
    with Reader(bag_dir) as reader:
        for conn, tstamp, raw in reader.messages():
            top = conn.topic
            if top == TCP:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                pt.append(tstamp); pv.append(np.asarray(m.data, dtype=np.float32))
            elif top == FIXED:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                ft.append(tstamp); fraw.append(bytes(m.data))
            elif top == GRIP:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                gt.append(tstamp); gv.append(bool(m.data))
    pt = np.array(pt); order = np.argsort(pt)
    pt, pv = pt[order], [pv[i] for i in order]
    return dict(pt=pt, pv=np.stack(pv), ft=np.array(ft), fraw=fraw, gt=np.array(gt), gv=gv)


def decode_native(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nearest(times, t):
    i = np.searchsorted(times, t)
    if i == 0: return 0
    if i >= len(times): return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def grip_hold(gt, gv, t):
    prev = False
    for te, ve in zip(gt, gv):
        if te <= t: prev = ve
        else: break
    return prev


def episode_frames(bag_dir):
    s = read_bag(bag_dir)
    t0 = max(s["pt"][0], s["ft"][0]); t1 = min(s["pt"][-1], s["ft"][-1])
    n = int((t1 - t0) / 1e9 * FS)
    grid = t0 + (np.arange(n) / FS * 1e9).astype(np.int64)
    tcp = np.stack([s["pv"][nearest(s["pt"], t)] for t in grid]).astype(np.float32)
    grip = np.array([grip_hold(s["gt"], s["gv"], t) for t in grid], dtype=np.float32)
    frames = []
    for t in range(n - 1):
        nat = decode_native(s["fraw"][nearest(s["ft"], grid[t])])
        frames.append(dict(
            img=cv2.resize(nat, (RES, RES), interpolation=cv2.INTER_AREA),
            state=tcp[t].copy(),
            action=np.concatenate([tcp[t + 1], grip[t + 1:t + 2]]).astype(np.float32),
        ))
    return frames


def main():
    # dim depuis la 1re collection
    first_eps = sorted(p for p in SRCS[0][1].glob("ep_*") if p.is_dir())
    dim = read_bag(first_eps[0])["pv"].shape[1]
    print(f"[cart-comb] pose TCP {dim}D, {RES}px, {int(FS)}Hz")
    root = Path("data_cache/lerobot_apple_cart_combined_128")
    if root.exists(): shutil.rmtree(root)
    feats = {
        "observation.images.fixed": {"dtype": "image", "shape": (3, RES, RES), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": [f"tcp_{i}" for i in range(dim)]},
        "action": {"dtype": "float32", "shape": (dim + 1,), "names": [f"tcp_{i}" for i in range(dim)] + ["gripper"]},
    }
    ds = LeRobotDataset.create(repo_id="local/apple_cart_combined_128", fps=int(FS), features=feats, root=root,
                               robot_type="real5dof", use_videos=False, image_writer_processes=0,
                               image_writer_threads=4, metadata_buffer_size=1)
    ep_idx = 0; total = 0
    for name, src in SRCS:
        eps = sorted(p for p in src.glob("ep_*") if p.is_dir())
        start = ep_idx
        for bag in eps:
            try:
                frames = episode_frames(bag)
            except Exception as e:
                print(f"[cart-comb] !! {name}/{bag.name} ECHEC: {e} -> skip"); continue
            for fr in frames:
                ds.add_frame({"observation.images.fixed": fr["img"],
                              "observation.state": fr["state"], "action": fr["action"], "task": TASK})
            ds.save_episode(); ep_idx += 1; total += len(frames)
        print(f"[cart-comb] SOURCE '{name}': épisodes index {start}..{ep_idx-1}  ({ep_idx-start} ép)")
    print(f"[cart-comb] TERMINE : {ds.meta.total_episodes} ép, {total} frames -> {root}")
    print(f"[cart-comb] >>> pour train/val : classique 0..N ; corrections à la fin (voir plages ci-dessus)")


if __name__ == "__main__":
    main()
