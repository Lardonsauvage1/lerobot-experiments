#!/usr/bin/env python
"""Courbe train+val loss vs step GLOBAL, run "pomme" (initial 0-2000 + prolongation 2000->...).
Paire chaque val_loss (valstep exact) avec la train loss loggee au meme is_log_step."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = ["results/logs/real/train.log", "results/logs/real/extend.log"]
OUT = Path("results/runs/real/apple_joint_224_r34/loss_evolution.png")

pts = {}  # step -> (train_loss, val_loss)
for lg in LOGS:
    p = Path(lg)
    if not p.exists():
        continue
    txt = p.read_text(errors="ignore").replace("\r", "\n")
    # tracker: ...loss:L grdn:... ; val: val_loss:V valstep:S. Ils se suivent (meme is_log_step).
    last_train = None
    for line in txt.split("\n"):
        m = re.search(r"loss:([0-9.]+) grdn:", line)
        if m:
            last_train = float(m.group(1))
        mv = re.search(r"val_loss:([0-9.]+) valstep:([0-9]+)", line)
        if mv:
            s = int(mv.group(2)); v = float(mv.group(1))
            pts[s] = (last_train, v)   # extend ecrase l'initial sur les steps communs

steps = sorted(pts)
tr = [pts[s][0] for s in steps]
vl = [pts[s][1] for s in steps]
# min val (ignore les None de train)
vmin_i = min(range(len(vl)), key=lambda i: vl[i]) if vl else None

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for a in ax:
    a.plot(steps, [t if t else float("nan") for t in tr], "-", color="#2563eb", lw=1.3, label="train loss")
    a.plot(steps, vl, "-o", color="#dc2626", ms=3, lw=1.3, label="val loss")
    if vmin_i is not None:
        a.plot(steps[vmin_i], vl[vmin_i], "*", color="#f59e0b", ms=18, markeredgecolor="k",
               label=f"min val {vl[vmin_i]:.4f} @ {steps[vmin_i]}")
    a.axvline(2000, color="gray", ls=":", lw=1, label="reprise (2000)")
    a.set_xlabel("step global"); a.grid(alpha=0.3)
ax[0].set_ylabel("loss"); ax[0].set_title("Échelle linéaire"); ax[0].legend(fontsize=8)
ax[1].set_yscale("log"); ax[1].set_title("Échelle log")
last = steps[-1] if steps else 0
fig.suptitle(f"Évolution de la loss — run « pomme » prolongé (step {last}, LR const 1e-4)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print(f"-> {OUT}  ({len(steps)} points, dernier step {last})")
if vmin_i is not None:
    print(f"min val = {vl[vmin_i]:.4f} @ step {steps[vmin_i]} ; val actuelle @ {steps[-1]} = {vl[-1]:.4f}")
