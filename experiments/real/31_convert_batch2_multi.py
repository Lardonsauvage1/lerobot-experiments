#!/usr/bin/env python
"""Convertit le batch 2 (79 ép, pomme) en 2 résolutions (96 + 128), caméra FIXE (left) uniquement,
EN IMAGES (use_videos=False, prêt gb10). Un seul passage sim (lit chaque mcap une fois).
Sorties : local/apple_b2_fixed_96 et local/apple_b2_fixed_128."""
import shutil
from pathlib import Path
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FS = 15.0
RES = [96, 128]
TASK = "pick the white apple and place it at the drop point"
SRC = Path("/Users/nielsmurawka/Documents/VScodeProject/NeuroneImitationCarote/batch_collect_20260716_023632")
TS = get_typestore(Stores.ROS2_JAZZY)
FIXED, JOINTS, GRIP = "/head_camera/left/image_raw/compressed", "/joint_states", "/gripper"


def read_bag(bag_dir):
    jt, jv, ft, fraw, gt, gv = [], [], [], [], [], []
    with Reader(bag_dir) as reader:
        for conn, tstamp, raw in reader.messages():
            top = conn.topic
            if top == JOINTS:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                jt.append(tstamp); jv.append(np.asarray(m.position, dtype=np.float32))
            elif top == FIXED:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                ft.append(tstamp); fraw.append(bytes(m.data))
            elif top == GRIP:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                gt.append(tstamp); gv.append(bool(m.data))
    jt = np.array(jt); order = np.argsort(jt)
    jt, jv = jt[order], [jv[i] for i in order]
    return dict(jt=jt, jv=np.stack(jv), ft=np.array(ft), fraw=fraw, gt=np.array(gt), gv=gv)


def decode_native(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # HWC uint8, natif 640x480


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
    t0 = max(s["jt"][0], s["ft"][0]); t1 = min(s["jt"][-1], s["ft"][-1])
    n = int((t1 - t0) / 1e9 * FS)
    grid = t0 + (np.arange(n) / FS * 1e9).astype(np.int64)
    joints = np.stack([s["jv"][nearest(s["jt"], t)] for t in grid]).astype(np.float32)
    grip = np.array([grip_hold(s["gt"], s["gv"], t) for t in grid], dtype=np.float32)
    frames = []
    for t in range(n - 1):
        nat = decode_native(s["fraw"][nearest(s["ft"], grid[t])])
        frames.append(dict(
            imgs={r: cv2.resize(nat, (r, r), interpolation=cv2.INTER_AREA) for r in RES},
            state=joints[t].copy(),
            action=np.concatenate([joints[t + 1], grip[t + 1:t + 2]]).astype(np.float32),
        ))
    return frames


def main():
    eps = sorted(p for p in SRC.glob("ep_*") if p.is_dir())
    nj = read_bag(eps[0])["jv"].shape[1]
    print(f"[b2-multi] {len(eps)} ép, {nj} joints, résolutions {RES}")
    dsets = {}
    for r in RES:
        root = Path(f"data_cache/lerobot_apple_b2_fixed_{r}")
        if root.exists(): shutil.rmtree(root)
        feats = {
            "observation.images.fixed": {"dtype": "image", "shape": (3, r, r), "names": ["channels", "height", "width"]},
            "observation.state": {"dtype": "float32", "shape": (nj,), "names": [f"joint_{i+1}" for i in range(nj)]},
            "action": {"dtype": "float32", "shape": (nj + 1,), "names": [f"jt_{i+1}" for i in range(nj)] + ["gripper"]},
        }
        dsets[r] = LeRobotDataset.create(repo_id=f"local/apple_b2_fixed_{r}", fps=int(FS), features=feats, root=root,
                                         robot_type="real5dof", use_videos=False, image_writer_processes=0,
                                         image_writer_threads=4, metadata_buffer_size=1)
    total = 0
    for bag in eps:
        try:
            frames = episode_frames(bag)
        except Exception as e:
            print(f"[b2-multi] !! {bag.name} ECHEC: {e} -> skip"); continue
        for fr in frames:
            for r in RES:
                dsets[r].add_frame({"observation.images.fixed": fr["imgs"][r],
                                    "observation.state": fr["state"], "action": fr["action"], "task": TASK})
        for r in RES:
            dsets[r].save_episode()
        total += len(frames)
        print(f"[b2-multi] {bag.name}: {len(frames)} frames (cumul {total})", flush=True)
    for r in RES:
        print(f"[b2-multi] TERMINE {r}px: {dsets[r].meta.total_episodes} ép, {total} frames")


if __name__ == "__main__":
    main()
