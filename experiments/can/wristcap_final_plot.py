#!/usr/bin/env python
"""Barplot FINAL WRISTCAP : le poignet ne bat JAMAIS l'agentview seule, à AUCUNE config
(capacité, résolution, augmentation). Les 3 paires matched sans/avec poignet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, succès, lo, hi, wrist)
DATA = [
    ("02\nagentview\nR18 96px",          70.4, 66.2, 74.2, False),
    ("08\n+poignet\nR18 96px",           55.2, 50.8, 59.5, True),
    ("A\nagentview\nR34+gros 96px",      67.4, 63.2, 71.4, False),
    ("B\n+poignet\nR34+gros 96px",       42.0, 37.8, 46.4, True),
    ("A'\nagentview\nR34+gros 224+crop", 72.4, 68.3, 76.1, False),
    ("B'\n+poignet\nR34+gros 224+crop",  35.2, 31.1, 39.5, True),
    ("A\"\nagentview\nSTANDARD DP 84", 96.0, 93.9, 97.4, False),
    ("B\"\n+poignet\nSTANDARD DP 84",  34.6, 30.6, 38.9, True),
]
fig, ax = plt.subplots(figsize=(13, 6))
x = range(len(DATA))
colors = ["#dc2626" if d[4] else "#2563eb" for d in DATA]
vals = [d[1] for d in DATA]
errs = [[d[1]-d[2] for d in DATA], [d[3]-d[1] for d in DATA]]
ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="k", linewidth=0.8)
for i, d in enumerate(DATA):
    ax.text(i, d[3]+1.2, f"{d[1]:.1f}", ha="center", fontweight="bold", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in DATA], fontsize=8.5)
ax.set_ylabel("Succès @500 rollouts (%)"); ax.set_ylim(0, 82)
# 4 paires matched : flèches sans->avec
for i in (0, 2, 4, 6):
    ax.annotate("", xy=(i+1, DATA[i+1][1]+3), xytext=(i, DATA[i][1]+3),
                arrowprops=dict(arrowstyle="->", color="#7c2d12", lw=1.8))
    d = DATA[i][1]-DATA[i+1][1]
    ax.text(i+0.5, max(DATA[i][1], DATA[i+1][1])+6, f"−{d:.0f}", color="#7c2d12",
            ha="center", fontweight="bold", fontsize=11)
ax.set_title("Can : le poignet (rouge) DÉGRADE l'agentview seule (bleu) à TOUTES les configs.\n"
             "La recette STANDARD DP corrige l'agentview (72->96%) mais le poignet crashe quand même (-61 pts)",
             fontsize=11.5, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2563eb", label="sans poignet"),
                   Patch(color="#dc2626", label="avec poignet")], loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT = "results/runs/can/wristcap_final_comparison.png"
fig.savefig(OUT, dpi=120)
print("->", OUT)
