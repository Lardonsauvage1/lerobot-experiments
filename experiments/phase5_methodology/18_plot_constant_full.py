"""Constant 1e-4 — succes 500-rollouts 1k->80k (toutes phases fusionnees).
Phases : 1-20k (initial) | 21-50k (continue) | 51-56k (continue2) | 57k (finevar, 1 pt)
         | 58-80k (resume). Montre le plateau bruite ~65% (pas de stabilisation au-dela).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
SRCS = [
    ROOT / "mini_constant" / "rollouts_500.csv",
    ROOT / "mini_constant_continue" / "rollouts_500.csv",
    ROOT / "mini_constant_continue2" / "rollouts_500.csv",
    ROOT / "mini_constant_resume" / "rollouts_500.csv",
]
rows = {}
for p in SRCS:
    for r in csv.DictReader(open(p)):
        rows[int(r["step"])] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
# 1 point de la zone dense finevar (57000) pour la continuite
for r in csv.DictReader(open(ROOT / "mini_constant_finevar" / "rollouts_500.csv")):
    if int(r["step"]) == 57000:
        rows[57000] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)

st = sorted(rows)
sr = [rows[s][0] for s in st]; lo = [rows[s][1] for s in st]; hi = [rows[s][2] for s in st]
plat = [v for s, v in zip(st, sr) if s >= 33000]
pmean = sum(plat)/len(plat)

fig, ax = plt.subplots(figsize=(13, 6))
ax.fill_between(st, lo, hi, color="tab:red", alpha=0.15, label="IC95 Wilson (n=500)")
ax.plot(st, sr, "-o", color="tab:red", ms=3.5)
ax.axhline(pmean, ls="--", c="gray", lw=1.2, label=f"plateau moyen (≥33k) = {pmean:.0f}%")
ax.axhline(2, ls=":", c="lightgray", lw=0.8)
for x, t in [(20000, "restart LR\n(annealé→0)"), (50000, "")]:
    ax.axvline(x, ls=":", c="steelblue", lw=1)
ax.text(20000, 6, " LR relancé", color="steelblue", fontsize=8)
ax.set_title("mini-CNN Can (vision pure) — constant 1e-4, succès 500-rollouts 1k→80k\n"
             f"plancher 2% (LR annealé) → plateau bruité ~{pmean:.0f}% (max {max(sr):.0f}%) ; ne se stabilise jamais finement",
             fontsize=12, weight="bold")
ax.set_xlabel("step"); ax.set_ylabel("succès (%)"); ax.set_ylim(-2, 85)
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "courbe_constant_1k_80k.png"
fig.savefig(out, dpi=150)
print(f"écrit : {out}")
print(f"plateau >=33k : moyenne {pmean:.1f}%, min {min(plat):.1f}%, max {max(plat):.1f}%, amplitude {max(plat)-min(plat):.1f} pts ({len(plat)} pts)")
