#!/usr/bin/env python
"""Graphe 4-panneaux du run PROPRE gb10 (dataset nettoyé, images) — sans rollout (pas de simulateur).
Panneaux : (A) train+val loss, (B) val-loss + min (juge offline), (C) grad-norm, (D) lr.
Enchaîne train 0-30k (LR const 1e-4) + cooldown 30k-35k (LR 1e-4->0), décalé de +30000.
Apparie chaque val_loss (valstep exact) avec la train-loss/grdn/lr du même log-step (log_freq=200).
Logs (copiés depuis gb10) : results/logs/real/gb10_train.log + gb10_cooldown.log."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("results/runs/real/apple_clean_gb10")
OUT = RUN / "gb10_clean_4panel.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
COOLDOWN_STEPS = 5000

def parse(logpath, offset=0):
    """-> liste de (step_global, train_loss, val_loss, grdn, lr)."""
    p = Path(logpath)
    if not p.exists():
        return []
    txt = p.read_text(errors="ignore").replace("\r", "\n")
    last = None  # (loss, grdn, lr)
    out = []
    for line in txt.split("\n"):
        mt = re.search(r"loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)", line)
        if mt:
            last = (float(mt.group(1)), float(mt.group(2)), float(mt.group(3)))
        mv = re.search(r"val_loss:([0-9.]+) valstep:([0-9]+)", line)
        if mv and last is not None:
            step = int(mv.group(2)) + offset
            out.append((step, last[0], float(mv.group(1)), last[1], last[2]))
    return out

train = parse("results/logs/real/gb10_train.log", offset=0)
cool = parse("results/logs/real/gb10_cooldown.log", offset=30000)
data = train + cool
if not data:
    raise SystemExit("aucun point parsé — vérifie les logs")

steps = [d[0] for d in data]
tloss = [d[1] for d in data]
vloss = [d[2] for d in data]
grdn = [d[3] for d in data]
lr = [d[4] for d in data]
vmin_i = min(range(len(vloss)), key=lambda i: vloss[i])
CDX = 30000  # frontière train/cooldown
OLD_BEST = 0.01069  # ancien déployable (cooldown@12000 dataset non nettoyé)

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Run PROPRE gb10 « prise de pomme » — dataset nettoyé (statiques rognés) — "
             "Diffusion R34+U-Net[128,256,512], 224px, joint 6D, LR const 1e-4 + cooldown 5k",
             fontsize=12, fontweight="bold")

def mark_cd(a):
    a.axvline(CDX, color="#94a3b8", ls="--", lw=1)
    a.axvspan(CDX, max(steps), color="#f1f5f9", alpha=0.6, zorder=0)

# (A) train + val
a = ax[0, 0]
a.plot(steps, tloss, "-", color="#2563eb", lw=1.3, label="train loss")
a.plot(steps, vloss, "-o", color="#dc2626", ms=3, lw=1.2, label="val loss")
mark_cd(a); a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log)")
a.set_title("(A) Loss train vs val  (zone grise = cooldown)"); a.legend(); a.grid(alpha=0.3)

# (B) val-loss + min + repère ancien best
b = ax[0, 1]
b.plot(steps, vloss, "-o", color="#dc2626", ms=4, lw=1.2)
b.axhline(OLD_BEST, color="#0891b2", ls=":", lw=1.5, label=f"ancien déployable = {OLD_BEST:.4f}")
b.plot(steps[vmin_i], vloss[vmin_i], "*", color="#f59e0b", ms=22, markeredgecolor="k",
       label=f"min val = {vloss[vmin_i]:.4f} @ {steps[vmin_i]}")
mark_cd(b); b.set_xlabel("step"); b.set_ylabel("val loss")
b.set_title("(B) Val-loss — JUGE offline"); b.legend(fontsize=9); b.grid(alpha=0.3)

# (C) grad norm
c = ax[1, 0]
c.plot(steps, grdn, "-", color="#059669", lw=1.1)
mark_cd(c); c.set_xlabel("step"); c.set_ylabel("grad norm"); c.set_title("(C) Grad-norm"); c.grid(alpha=0.3)

# (D) lr
d = ax[1, 1]
d.plot(steps, lr, "-", color="#7c3aed", lw=1.5)
mark_cd(d); d.set_xlabel("step"); d.set_ylabel("lr")
d.set_title("(D) Learning rate (const 1e-4 -> cooldown 0)"); d.grid(alpha=0.3)
d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=120)
print(f"-> {OUT}")
print(f"points: train={len(train)} cooldown={len(cool)} | MIN val-loss = {vloss[vmin_i]:.4f} @ step {steps[vmin_i]}")
print(f"ancien déployable (dataset non nettoyé) = {OLD_BEST:.4f} -> "
      f"{'AMÉLIORÉ' if vloss[vmin_i] < OLD_BEST else 'PAS mieux'} "
      f"({100*(vloss[vmin_i]-OLD_BEST)/OLD_BEST:+.1f}%)")
