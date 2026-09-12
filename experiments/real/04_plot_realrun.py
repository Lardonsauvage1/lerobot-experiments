#!/usr/bin/env python
"""Graphe du run reel "pomme" — variante 4-panneaux SANS succes (pas de simulateur).
Panneaux : (A) train+val loss, (B) val-loss avec min (notre juge), (C) grad-norm, (D) lr.
Source = results/logs/real/train.log (tqdm \r nettoye)."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("results/logs/real/train.log")
OUT = Path("results/runs/real/apple_joint_224_r34/realrun_4panel.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

txt = LOG.read_text(errors="ignore").replace("\r", "\n")

# tracker lines : step:NK ... loss:X grdn:Y lr:Z  (dans l'ordre, log_freq=500)
tr = re.findall(r"loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)", txt)
steps = [500 * (i + 1) for i in range(len(tr))]
tloss = [float(a) for a, _, _ in tr]
grdn = [float(b) for _, b, _ in tr]
lr = [float(c) for _, _, c in tr]

# val_loss : exact valstep
vl = re.findall(r"val_loss:([0-9.]+) valstep:([0-9]+)", txt)
vsteps = [int(s) for _, s in vl]
vloss = [float(v) for v, _ in vl]
vmin_i = min(range(len(vloss)), key=lambda i: vloss[i]) if vloss else None

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Run réel « prise de pomme » — Diffusion R34 + U-Net[128,256,512], 224px, "
             f"joint absolu 6D, LR const 1e-4 + EMA  (step {steps[-1] if steps else 0}/30000)",
             fontsize=12, fontweight="bold")

# (A) train + val loss
a = ax[0, 0]
a.plot(steps, tloss, "-", color="#2563eb", lw=1.5, label="train loss")
a.plot(vsteps, vloss, "-o", color="#dc2626", ms=4, lw=1.5, label="val loss")
a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log)")
a.set_title("(A) Loss train vs val"); a.legend(); a.grid(alpha=0.3)

# (B) val-loss avec min (le juge offline)
b = ax[0, 1]
b.plot(vsteps, vloss, "-o", color="#dc2626", ms=5, lw=1.5)
if vmin_i is not None:
    b.plot(vsteps[vmin_i], vloss[vmin_i], "*", color="#f59e0b", ms=22,
           markeredgecolor="k", label=f"min val = {vloss[vmin_i]:.4f} @ {vsteps[vmin_i]}")
    b.legend()
b.set_xlabel("step"); b.set_ylabel("val loss")
b.set_title("(B) Val-loss — JUGE offline (pas de rollout)"); b.grid(alpha=0.3)

# (C) grad norm
c = ax[1, 0]
c.plot(steps, grdn, "-", color="#059669", lw=1.2)
c.set_xlabel("step"); c.set_ylabel("grad norm"); c.set_title("(C) Grad-norm"); c.grid(alpha=0.3)

# (D) lr
d = ax[1, 1]
d.plot(steps, lr, "-", color="#7c3aed", lw=1.5)
d.set_xlabel("step"); d.set_ylabel("lr"); d.set_title("(D) Learning rate (constant 1e-4)")
d.grid(alpha=0.3); d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=120)
print(f"-> {OUT}")
print(f"train steps: {len(steps)} | val points: {len(vloss)}")
if vloss:
    print("val_loss:", {vsteps[i]: vloss[i] for i in range(len(vloss))})
    print(f"MIN val-loss = {vloss[vmin_i]:.4f} @ step {vsteps[vmin_i]}")
