#!/usr/bin/env python3
"""Compare la convergence du joint BRUT vs EMA sur la plage du push (50k->80k).
Lit rollouts_50.csv des deux run-dirs (joint_r34_bigunet et _ema), trace les deux
courbes succès + IC95 sur la plage commune.
  venv312/bin/python experiments/can/64_plot_raw_vs_ema.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results/runs/can/joint_r34_bigunet/rollouts_50.csv"
EMA = "results/runs/can/joint_r34_bigunet_ema/rollouts_50.csv"
OUT = "results/runs/can/joint_r34_bigunet_ema/raw_vs_ema_50_80k.png"
SMIN = 50000  # on ne compare que la plage du push


def load(path):
    by = {}
    for r in csv.DictReader(open(path)):
        by[int(float(r["step"]))] = r
    g = lambda r, k, d=None: float(r[k]) if r.get(k) not in (None, "") else d
    st = sorted(by)
    sr = np.array([g(by[s], "success_rate") * 100 for s in st])
    lo = np.array([g(by[s], "ci95_low", g(by[s], "success_rate")) * 100 for s in st])
    hi = np.array([g(by[s], "ci95_high", g(by[s], "success_rate")) * 100 for s in st])
    st = np.array(st)
    m = st >= SMIN
    return st[m], sr[m], lo[m], hi[m]


fig, ax = plt.subplots(figsize=(9, 5.5))
for path, color, label in [(RAW, "#1f77b4", "BRUT (poids SGD)"), (EMA, "#d62728", "EMA (decay 0.9999)")]:
    try:
        st, sr, lo, hi = load(path)
        if len(st):
            ax.fill_between(st / 1000, lo, hi, color=color, alpha=0.15)
            ax.plot(st / 1000, sr, "-o", color=color, ms=4, lw=1.8, label=f"{label}")
            best = st[int(np.argmax(sr))]
            ax.annotate(f"max {sr.max():.0f}% @{best//1000}k", (best / 1000, sr.max()),
                        color=color, fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
    except FileNotFoundError:
        print(f"absent: {path}")

ax.set_xlabel("step (k)"); ax.set_ylabel("succès %"); ax.set_ylim(-3, 103)
ax.set_title("Joint poussé 50k→80k — BRUT vs EMA (n=50, kp=50)")
ax.legend(loc="best"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT, dpi=120)
print(f"Sauvé : {OUT}")
