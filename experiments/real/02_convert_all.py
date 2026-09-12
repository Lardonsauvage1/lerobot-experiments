#!/usr/bin/env python
"""Convertit les 138 bags mcap ROS2 (tache pomme) en un LeRobotDataset `local/apple_joint_224`.

- Resample 15 Hz, alignement nearest-timestamp (valide par 01_witness_episode.py : residu ~4 ms).
- 2 cameras JPEG 640x480 -> RGB 224x224 (observation.images.left / .right).
- observation.state = joints(t) [5] ; action = [joints(t+1)[5], gripper(t+1)[1]] (open-loop, absolu 6D).
- gripper : hold de la derniere valeur, defaut False (ouvert) avant le 1er event.
- On drope la derniere frame de chaque episode (pas de t+1).

Split val gere a l'entrainement (episodes hors --dataset.episodes).
"""
import sys, shutil
from pathlib import Path
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FS = 15.0
IMG = 224
TASK = "pick the white apple and place it at the drop point"
SRC = Path("/Users/nielsmurawka/Documents/VScodeProject/NeuroneImitationCarote/Demis_dataset/batch_689428411_20260712_025841")
REPO = "local/apple_joint_224"
ROOT = Path("data_cache/lerobot_apple_joint_224")
TS = get_typestore(Stores.ROS2_JAZZY)
LEFT, RIGHT, JOINTS, GRIP = ("/head_camera/left/image_raw/compressed",
                             "/head_camera/right/image_raw/compressed",
                             "/joint_states", "/gripper")


def read_bag(bag_dir):
    jt, jv, lt, lraw, rt, rraw, gt, gv = [], [], [], [], [], [], [], []
    with Reader(bag_dir) as reader:
        for conn, tstamp, raw in reader.messages():
            top = conn.topic
            if top == JOINTS:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                jt.append(tstamp); jv.append(np.asarray(m.position, dtype=np.float32))
            elif top == LEFT:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                lt.append(tstamp); lraw.append(bytes(m.data))
            elif top == RIGHT:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                rt.append(tstamp); rraw.append(bytes(m.data))
            elif top == GRIP:
                m = TS.deserialize_cdr(raw, conn.msgtype)
                gt.append(tstamp); gv.append(bool(m.data))
    jt = np.array(jt); order = np.argsort(jt)
    jt, jv = jt[order], [jv[i] for i in order]
    return dict(jt=jt, jv=np.stack(jv), lt=np.array(lt), lraw=lraw,
                rt=np.array(rt), rraw=rraw, gt=np.array(gt), gv=gv)


def decode(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # HWC uint8


def nearest(times, t):
    i = np.searchsorted(times, t)
    if i == 0: return 0
    if i >= len(times): return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i-1] - t) else i - 1


def grip_hold(gt, gv, t):
    prev = False
    for te, ve in zip(gt, gv):
        if te <= t: prev = ve
        else: break
    return prev


def episode_frames(bag_dir):
    s = read_bag(bag_dir)
    t0 = max(s["jt"][0], s["lt"][0], s["rt"][0])
    t1 = min(s["jt"][-1], s["lt"][-1], s["rt"][-1])
    n = int((t1 - t0) / 1e9 * FS)
    grid = t0 + (np.arange(n) / FS * 1e9).astype(np.int64)
    joints = np.stack([s["jv"][nearest(s["jt"], t)] for t in grid]).astype(np.float32)  # [n,5]
    grip = np.array([grip_hold(s["gt"], s["gv"], t) for t in grid], dtype=np.float32)   # [n]
    # action(t) = pose suivante -> on garde n-1 frames
    frames = []
    for t in range(n - 1):
        frames.append(dict(
            left=decode(s["lraw"][nearest(s["lt"], grid[t])]),
            right=decode(s["rraw"][nearest(s["rt"], grid[t])]),
            state=joints[t].copy(),                                   # [5]
            action=np.concatenate([joints[t + 1], grip[t + 1:t + 2]]).astype(np.float32),  # [6]
        ))
    return frames


def main():
    eps = sorted(p for p in SRC.glob("ep_*") if p.is_dir())
    print(f"[convert] {len(eps)} episodes trouves")
    if ROOT.exists():
        print(f"[convert] suppression ancien {ROOT}"); shutil.rmtree(ROOT)

    features = {
        "observation.images.left":  {"dtype": "video", "shape": (3, IMG, IMG), "names": ["channels", "height", "width"]},
        "observation.images.right": {"dtype": "video", "shape": (3, IMG, IMG), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (5,), "names": [f"joint_{i+1}" for i in range(5)]},
        "action": {"dtype": "float32", "shape": (6,), "names": [f"jt_{i+1}" for i in range(5)] + ["gripper"]},
    }
    ds = LeRobotDataset.create(repo_id=REPO, fps=int(FS), features=features,
                               root=ROOT, robot_type="real5dof", use_videos=True,
                               image_writer_processes=0, image_writer_threads=4)

    total = 0
    for k, bag in enumerate(eps):
        try:
            frames = episode_frames(bag)
        except Exception as e:
            print(f"[convert] !! {bag.name} ECHEC: {e} -> skip")
            continue
        for fr in frames:
            ds.add_frame({
                "observation.images.left": fr["left"],
                "observation.images.right": fr["right"],
                "observation.state": fr["state"],
                "action": fr["action"],
                "task": TASK,
            })
        ds.save_episode()
        total += len(frames)
        print(f"[convert] {bag.name}: {len(frames)} frames (cumul {total})", flush=True)

    print(f"[convert] TERMINE: {ds.meta.total_episodes} episodes, {total} frames -> {ROOT}")


if __name__ == "__main__":
    main()
