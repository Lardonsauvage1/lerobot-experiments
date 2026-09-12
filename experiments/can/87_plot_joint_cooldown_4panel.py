"""4-PANNEAUX joint ENRICHI cooldown : succès (brut tous ckpts + merge SWA + cooldown LR=0 + late10)
/ loss / lr / grad. Synthèse complète : le brut rebondit, le merge lisse, le cooldown monte ~83 % stable.
  venv312/bin/python experiments/can/87_plot_joint_cooldown_4panel.py
"""
import csv, glob, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/can"
RD = f"{R}/joint_r34_bigunet"
LOG_GLOB = "results/logs/can/run_joint_full*.log results/logs/can/run_joint_ext.log"

def smooth(y, w=15):
    y = np.asarray(y, float)
    if len(y) < 3: return y
    h = w // 2
    return np.array([y[max(0, i-h):min(len(y), i+h+1)].mean() for i in range(len(y))])

def load_log(globs):
    steps, tl, vl_, gr, lr = [], [], [], [], []
    seen = set()
    files = []
    for g in globs.split(): files += glob.glob(g)
    for lg in sorted(set(files)):
        txt = open(lg, errors="ignore").read()
        tr = re.findall(r"(?<!val_)loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)", txt)
        vl = re.findall(r"val_loss:([0-9.]+) valstep:(\d+)", txt)
        for i in range(min(len(tr), len(vl))):
            s = int(vl[i][1])
            if s in seen: continue
            seen.add(s); steps.append(s)
            tl.append(float(tr[i][0])); vl_.append(float(vl[i][0]))
            gr.append(float(tr[i][1])); lr.append(float(tr[i][2]))
    o = np.argsort(steps)
    return [np.array(x)[o] for x in (steps, tl, vl_, gr, lr)] if steps else [np.array([])]*5

def col(path, k=3):  # success_rate% d'un r.csv (col 3)
    rows = list(csv.DictReader(open(path)))
    return float(rows[-1]["success_rate"])*100 if rows else None

# --- données panneau succès ---
raw = {int(float(r["step"])): float(r["success_rate"])*100 for r in csv.DictReader(open(f"{RD}/rollouts_50.csv"))}
rx = sorted(raw); ry = [raw[s] for s in rx]

WIN = {20:"joint_swa_W1_12_20k",30:"joint_swa_W2_22_30k",40:"joint_swa_W3_32_40k",
       50:"joint_swa_W4_42_50k",60:"joint_swa_e56_52_60k",70:"joint_swa_e66_62_70k",80:"joint_swa_e76_72_80k"}
mdict = {}
for k, d in WIN.items():
    p = f"{R}/{d}/r.csv"
    if Path(p).exists(): mdict[k] = col(p)
mp = f"{R}/merge_profile.csv"   # merges denses prioritaires
if Path(mp).exists():
    for r in csv.DictReader(open(mp)): mdict[int(float(r["source_k"]))] = float(r["success_rate"])*100
mx = sorted(mdict); my = [mdict[k] for k in mx]

cd = {}
if Path(f"{R}/cooldown_profile.csv").exists():
    for r in csv.DictReader(open(f"{R}/cooldown_profile.csv")):
        cd[int(float(r["source_k"]))] = float(r["success_rate"])*100
cx = sorted(cd); cy = [cd[k] for k in cx]

lst, tl, vl, gr, lr = load_log(LOG_GLOB)

# --- figure ---
fig, ax = plt.subplots(2, 2, figsize=(14, 9)); ax = ax.flatten()
# panneau 0 : succès — 3 niveaux
ax[0].plot(np.array(rx)/1000, ry, "-^", color="#b0b0b0", ms=5, lw=1.2, alpha=0.9,
           label="brut — tous checkpoints (n=50, le REBOND)")
ax[0].plot(mx, my, "-s", color="#d62728", ms=7, lw=1.8, label="merge SWA — fenêtre 5 ckpts")
ax[0].plot(cx, cy, "-o", color="#1f2d3d", ms=9, lw=2.4, zorder=6, label="cooldown LR=0 (annealing, n=100)")
for k in cx: ax[0].annotate(f"{cd[k]:.0f}", (k, cd[k]), color="#1f2d3d", fontsize=8.5, fontweight="bold",
                            xytext=(0, 8), textcoords="offset points", ha="center")
ax[0].axhline(81.6, color="#2ca02c", ls="--", lw=1.3, alpha=0.7, label="SWA late10 = 81,6 % @500")
ax[0].set_title("Succès (rollouts) % — brut → merge → cooldown"); ax[0].set_ylim(-3, 102)
ax[0].grid(alpha=.3); ax[0].set_xlabel("step / checkpoint de départ (k)"); ax[0].legend(loc="lower right", fontsize=8)
# 1) loss
if len(lst):
    ax[1].plot(lst/1000, smooth(tl), "-", color="#f4a3a3", lw=1.3, label="train (lissée)")
    ax[1].plot(lst/1000, smooth(vl), "-", color="#d62728", lw=2, label="val (lissée)")
    ax[1].set_yscale("log"); ax[1].set_title("loss train + val — log")
    ax[1].grid(alpha=.3, which="both"); ax[1].set_xlabel("step (k)"); ax[1].legend(fontsize=8)
    ax[2].plot(lst/1000, lr, "-", color="#d62728", lw=1.5); ax[2].set_title("learning rate (constant 1e-4)")
    ax[2].grid(alpha=.3); ax[2].set_xlabel("step (k)")
    ax[3].plot(lst/1000, gr, "-", color="#d62728", lw=0.7, alpha=.5)
    ax[3].plot(lst/1000, smooth(gr), "-", color="#d62728", lw=2)
    ax[3].set_yscale("log"); ax[3].set_title("grad_norm — log")
    ax[3].grid(alpha=.3, which="both"); ax[3].set_xlabel("step (k)")
fig.suptitle("Joint ResNet34 (LR constant) — 4 panneaux + cooldown : le brut rebondit (0-56%), "
             "le merge lisse (~70), le cooldown monte ~83% stable", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); out = f"{RD}/full_curves_cooldown.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, f"| brut {len(rx)} ckpts, merge {len(mx)}, cooldown {len(cx)}, log {len(lst)} pts")
