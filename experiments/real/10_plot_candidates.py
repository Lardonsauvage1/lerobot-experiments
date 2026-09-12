#!/usr/bin/env python
"""Barplot des candidats "prise de pomme" (val-loss) a partir de compare_valloss.json."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("results/runs/real/apple_joint_224_r34")
d = json.loads((RUN / "compare_valloss.json").read_text())
rk = d["ranking"]
names = [r["name"] for r in rk]
vals = [r["val_loss"] for r in rk]
errs = [r["std"] for r in rk]
colors = ["#059669" if i == 0 else "#64748b" for i in range(len(rk))]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.barh(range(len(rk)), vals, xerr=errs, color=colors, edgecolor="k", capsize=4)
ax.set_yticks(range(len(rk))); ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("val-loss (moyenne sur ép. 40-45, plus bas = mieux)")
ax.set_title(f"Candidats modèle déployable « pomme » — DÉPLOYABLE = {d['deployable']}", fontweight="bold")
for i, (v, e) in enumerate(zip(vals, errs)):
    ax.text(v + e + max(vals) * 0.01, i, f"{v:.4f}", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = RUN / "candidates_valloss.png"
fig.savefig(out, dpi=120)
print(f"-> {out}")
