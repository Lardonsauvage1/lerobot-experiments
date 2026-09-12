#!/usr/bin/env python
"""Barplot WRISTCAP : le poignet aide-t-il ? Comparaison sans/avec poignet à toutes capacités.
Résultat : le poignet ne dépasse JAMAIS l'agentview seule, et à haute capacité il DÉGRADE (67->42)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, succès %, IC95 lo, hi, avec_poignet, capacité)
DATA = [
    ("02\nagentview\nR18",              70.4, 66.2, 74.2, False, "R18/petit"),
    ("A (cd)\nagentview\nR34+gros",     67.4, 63.2, 71.4, False, "R34/gros"),
    ("08\n+poignet\nR18 sép",           55.2, 50.8, 59.5, True,  "R18/petit"),
    ("14\n+poignet\nR18 sép+gros",      68.0, 63.8, 72.0, True,  "R18/gros"),
    ("B (cd)\n+poignet\nR34+gros sép",  42.0, 37.8, 46.4, True,  "R34/gros"),
]
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(DATA))
colors = ["#2563eb" if not d[4] else "#dc2626" for d in DATA]
vals = [d[1] for d in DATA]
errs = [[d[1]-d[2] for d in DATA], [d[3]-d[1] for d in DATA]]
bars = ax.bar(x, vals, yerr=errs, capsize=6, color=colors, edgecolor="k", linewidth=0.8)
for i, d in enumerate(DATA):
    ax.text(i, d[3]+1.2, f"{d[1]:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in DATA], fontsize=9)
ax.set_ylabel("Succès @500 rollouts (%)"); ax.set_ylim(0, 82)
ax.axhline(70.4, color="#2563eb", ls=":", lw=1, alpha=0.6)
ax.set_title("Can vision-pure : le poignet (rouge) ne bat JAMAIS l'agentview seule (bleu)\n"
             "et à HAUTE capacité (A vs B, paire matched) il DÉGRADE : 67,4 % → 42,0 %",
             fontsize=12, fontweight="bold")
# flèche A->B
ax.annotate("", xy=(4, 45), xytext=(1, 64),
            arrowprops=dict(arrowstyle="->", color="#7c2d12", lw=2))
ax.text(2.5, 58, "+ poignet\nà capacité égale\n−25 pts", color="#7c2d12", fontsize=10,
        ha="center", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2563eb", label="sans poignet"), Patch(color="#dc2626", label="avec poignet")],
          loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT = "results/runs/can/wristcap_comparison.png"
fig.savefig(OUT, dpi=120)
print("->", OUT)
