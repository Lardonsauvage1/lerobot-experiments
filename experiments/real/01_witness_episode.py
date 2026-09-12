#!/usr/bin/env python
"""Convertisseur TEMOIN — 1 episode mcap ROS2 -> frames LeRobot-like, avec visuels de controle.

But : valider l'alignement (resample 15 Hz nearest-timestamp), le decodage JPEG, le hold du
gripper et la coherence des 5 joints AVANT de convertir les 138 episodes.

Sortie : results/runs/real/witness/ep_000_check.png (montage images + courbes joints/gripper).
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

FS = 15.0                       # cadence commune (Hz) — decision user
IMG_SIZE = 224                  # resize final
TS = get_typestore(Stores.ROS2_JAZZY)
LEFT = "/head_camera/left/image_raw/compressed"
RIGHT = "/head_camera/right/image_raw/compressed"
JOINTS = "/joint_states"
GRIP = "/gripper"


def read_bag(bag_dir):
    """Retourne les flux bruts tries par timestamp (en secondes, origine = 1er msg)."""
    jt, jv = [], []            # joints : time, [5]
    lt, lraw = [], []          # left : time, jpeg bytes
    rt, rraw = [], []          # right
    gt, gv = [], []            # gripper events : time, bool
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
    jt = np.array(jt); lt = np.array(lt); rt = np.array(rt); gt = np.array(gt)
    order = np.argsort(jt); jt, jv = jt[order], [jv[i] for i in order]
    return dict(jt=jt, jv=np.stack(jv), lt=lt, lraw=lraw, rt=rt, rraw=rraw, gt=gt, gv=gv)


def decode_resize(jpeg_bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR 480x640x3
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nearest_idx(times, t):
    i = np.searchsorted(times, t)
    if i == 0: return 0
    if i >= len(times): return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i-1] - t) else i - 1


def gripper_hold(gt, gv, t):
    """Derniere valeur <= t ; avant le 1er event -> False (ouvert, pre-arme)."""
    prev = False
    for te, ve in zip(gt, gv):
        if te <= t: prev = ve
        else: break
    return prev


def resample(streams, fs=FS):
    """Grille commune sur le recouvrement des 3 flux, a fs Hz."""
    t0 = max(streams["jt"][0], streams["lt"][0], streams["rt"][0])
    t1 = min(streams["jt"][-1], streams["lt"][-1], streams["rt"][-1])
    n = int((t1 - t0) / 1e9 * fs)
    grid = t0 + (np.arange(n) / fs * 1e9).astype(np.int64)
    joints = np.stack([streams["jv"][nearest_idx(streams["jt"], t)] for t in grid])
    grip = np.array([gripper_hold(streams["gt"], streams["gv"], t) for t in grid], dtype=np.float32)
    li = [nearest_idx(streams["lt"], t) for t in grid]
    ri = [nearest_idx(streams["rt"], t) for t in grid]
    return grid, joints, grip, li, ri


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/nielsmurawka/Documents/VScodeProject/NeuroneImitationCarote/Demis_dataset/batch_689428411_20260712_025841/ep_000"
    out = Path("results/runs/real/witness"); out.mkdir(parents=True, exist_ok=True)

    print(f"[witness] lecture {bag}")
    s = read_bag(bag)
    grid, joints, grip, li, ri = resample(s)
    T = len(grid)
    tsec = (grid - grid[0]) / 1e9
    print(f"[witness] {T} frames a {FS} Hz ({tsec[-1]:.2f}s)")
    print(f"[witness] joints range: min={joints.min(0)}, max={joints.max(0)}")
    print(f"[witness] gripper transitions (frame): "
          f"{[i for i in range(1, T) if grip[i] != grip[i-1]]} -> valeurs {grip[[0,-1]]}")

    # nearest-alignment residuals (ms) : ecart entre le timestamp grille et le msg choisi
    res_l = np.array([abs(s["lt"][li[k]] - grid[k]) for k in range(T)]) / 1e6
    res_r = np.array([abs(s["rt"][ri[k]] - grid[k]) for k in range(T)]) / 1e6
    print(f"[witness] residu alignement image (ms) L: med={np.median(res_l):.1f} max={res_l.max():.1f}"
          f" | R: med={np.median(res_r):.1f} max={res_r.max():.1f}")

    # --- visuel de controle ---
    picks = np.linspace(0, T - 1, 5).astype(int)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 5, height_ratios=[2, 2, 3])
    for c, k in enumerate(picks):
        axl = fig.add_subplot(gs[0, c]); axl.imshow(decode_resize(s["lraw"][li[k]])); axl.axis("off")
        axl.set_title(f"L t={tsec[k]:.1f}s grip={'C' if grip[k] else 'O'}", fontsize=9)
        axr = fig.add_subplot(gs[1, c]); axr.imshow(decode_resize(s["rraw"][ri[k]])); axr.axis("off")
        axr.set_title(f"R t={tsec[k]:.1f}s", fontsize=9)
    axj = fig.add_subplot(gs[2, :])
    for d in range(5):
        axj.plot(tsec, joints[:, d], label=f"joint_{d+1}")
    axj.plot(tsec, grip * joints.max() * 0.9, "k--", lw=1.5, label="gripper (C=haut)")
    axj.set_xlabel("t (s)"); axj.set_ylabel("rad"); axj.legend(ncol=6, fontsize=8)
    axj.set_title(f"ep_000 — 5 joints + gripper @ {FS} Hz  ({T} frames)")
    fig.tight_layout()
    p = out / "ep_000_check.png"; fig.savefig(p, dpi=110); print(f"[witness] -> {p}")

    # dump un petit resume machine-lisible
    (out / "ep_000_summary.json").write_text(json.dumps(dict(
        frames=T, fs=FS, dur_s=float(tsec[-1]),
        joint_min=joints.min(0).tolist(), joint_max=joints.max(0).tolist(),
        grip_transitions=[i for i in range(1, T) if grip[i] != grip[i-1]],
        align_resid_ms=dict(left_med=float(np.median(res_l)), left_max=float(res_l.max()),
                            right_med=float(np.median(res_r)), right_max=float(res_r.max())),
    ), indent=2))


if __name__ == "__main__":
    main()
