#!/usr/bin/env python3
"""Courbe du COOLDOWN joint (anneal LR 1e-4 -> 0 sur 5k depuis le plateau constant 50k).
Succès (rollouts n=50) vs step + IC95, axe LR qui descend, références SWA (81,6%) et plateau brut (~76%).
  venv312/bin/python experiments/can/83_plot_cooldown.py
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "results/runs/can/joint_cooldown_50k"
LR0, NCD = 1e-4, 5000  # cooldown : LR0 -> 0 lineaire sur NCD steps

xs, ys, lo, hi = [], [], [], []
for r in csv.DictReader(open(f"{RUN}/rollouts_50.csv")):
    s = int(float(r["step"]))
    xs.append(s); ys.append(float(r["success_rate"])*100)
    lo.append((float(r["success_rate"]) - float(r["ci95_low"]))*100)
    hi.append((float(r["ci95_high"]) - float(r["success_rate"]))*100)

fig, ax = plt.subplots(figsize=(10, 6.5))

# references
ax.axhline(81.6, color="#27ae60", ls="--", lw=1.6, label="SWA late10 (merge) = 81,6 % @500")
ax.axhline(76,   color="#e67e22", ls=":",  lw=1.6, label="plateau brut constant ≈ 76 %")

# courbe cooldown
ax.errorbar(xs, ys, yerr=[lo, hi], color="#2c3e50", marker="o", ms=8, lw=2.2,
            capsize=5, zorder=5, label="cooldown (succès, n=50, IC95)")
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.0f}%", (x, y), color="#2c3e50", fontsize=10, fontweight="bold",
                xytext=(0, 9), textcoords="offset points", ha="center")
# souligne le point final (LR=0)
ax.annotate("LR=0\nmodèle posé", (xs[-1], ys[-1]), color="#c0392b", fontsize=9, fontweight="bold",
            xytext=(-52, -2), textcoords="offset points", ha="center")

# axe LR (descend vers 0)
ax2 = ax.twinx()
lr_x = list(range(0, NCD+1, 250))
lr_y = [LR0*max(0, 1 - s/NCD) for s in lr_x]
ax2.plot(lr_x, lr_y, "-", color="#3498db", lw=1.4, alpha=0.6)
ax2.set_ylabel("learning rate (cooldown linéaire)", color="#3498db")
ax2.tick_params(axis="y", labelcolor="#3498db"); ax2.set_ylim(0, LR0*1.15)
ax2.annotate("LR 1e-4 → 0", (1200, LR0*0.82), color="#3498db", fontsize=9)

ax.set_xlabel("step de cooldown (depuis le plateau constant 50k)", fontsize=12)
ax.set_ylabel("succès (rollouts Can) %", fontsize=12)
ax.set_ylim(0, 100); ax.set_xlim(0, NCD+200); ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Joint — COOLDOWN depuis 50k : le succès grimpe à mesure que le LR → 0\n"
             "le checkpoint posé (5k, LR=0) = 92 % > SWA 81,6 % → l'archi n'était PAS maxée (n=50, à confirmer @500)",
             fontsize=11)
fig.tight_layout(); out = f"{RUN}/cooldown_curve.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, "| points :", ", ".join(f"{x//1000}k={y:.0f}%" for x, y in zip(xs, ys)))
