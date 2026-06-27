#!/usr/bin/env python3
"""Effet de la LARGEUR de la fenêtre de merge (SWA) sur le fond — mini-CNN Can.
À fin fixe, on moyenne 5 / 10 / 15 checkpoints. Teste le levier dominant de WSM (durée du merge).
3 ancrages : constant @150k, cosine @150k, cosine @250k. Lit river_{const,cos}_{L,M,E}{5,10,15}/r.csv.
  venv312/bin/python experiments/phase5_methodology/35_plot_windowsize_minicnn.py
"""
import csv, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/phase5_methodology"

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n; c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-h)*100, min(1.,c+h)*100

def floor(d):
    p = f"{R}/{d}/r.csv"
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    r = rows[-1]; rate = float(r["success_rate"])*100
    lo = float(r.get("ci95_low") or 0)*100; hi = float(r.get("ci95_high") or 0)*100
    n = int(float(r.get("n") or 500))
    if hi <= lo: lo, hi = wilson(round(rate/100*n), n)
    return rate, lo, hi

SIZES = [5, 10, 15]
ANCHORS = [
    ("constant @150k", "#c0392b", ["river_const_L5", "river_const_L10", "river_const_L15"]),
    ("cosine @150k",   "#1f4e8c", ["river_cos_M5",   "river_cos_M10",   "river_cos_M15"]),
    ("cosine @250k",   "#27ae60", ["river_cos_E5",   "river_cos_E10",   "river_cos_E15"]),
]

fig, ax = plt.subplots(figsize=(9, 6.5))
for lab, col, dirs in ANCHORS:
    xs, ys, los, his = [], [], [], []
    for sz, d in zip(SIZES, dirs):
        res = floor(d)
        if res is None: continue
        f, lo, hi = res; xs.append(sz); ys.append(f); los.append(f-lo); his.append(hi-f)
    if xs:
        ax.errorbar(xs, ys, yerr=[los, his], color=col, marker="o", ms=7, lw=2,
                    capsize=5, label=lab)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), color=col, fontsize=9, fontweight="bold",
                        xytext=(6, 4), textcoords="offset points")

ax.set_xticks(SIZES); ax.set_xlabel("largeur de la fenêtre de merge (nb checkpoints moyennés)", fontsize=12)
ax.set_ylabel("fond SWA — succès (rollouts Can) %", fontsize=12)
ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=10)
ax.set_title("mini-CNN Can — effet de la LARGEUR du merge (SWA), à fin fixe\n"
             "moyenner plus de checkpoints élève-t-il le fond ? (levier dominant WSM)", fontsize=11.5)
fig.tight_layout(); out = f"{R}/courbes_river_minicnn_windowsize.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, "|",
      ", ".join(f"{lab}: " + "/".join(f"{floor(d)[0]:.0f}" if floor(d) else "?" for d in dirs)
               for lab, _, dirs in ANCHORS))
