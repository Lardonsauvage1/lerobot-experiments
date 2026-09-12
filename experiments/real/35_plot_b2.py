#!/usr/bin/env python
"""4-panneaux DÉTAILLÉS (log_freq=50 -> ~600 points) des modèles apple b2 (1 cam fixe).
Pas de rollout (pas de simulateur pomme) -> panneaux : (A) train+val loss, (B) val-loss+min, (C) grad, (D) lr.
Enchaîne train 0-30k + cooldown 30k-35k (décalé). Usage : python 35_plot_b2.py 96|128."""
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "96"
TRAIN = Path(f"results/logs/real/apple_b2_{RES}.log")
COOL = Path(f"results/logs/real/cd_apple_b2_{RES}.log")
OUT = Path(f"results/runs/real/apple_b2_fixed_{RES}/b2_{RES}_4panel.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def parse(p, offset=0):
    if not p.exists():
        return []
    txt = p.read_text(errors="ignore").replace("\r", "\n")
    last, out = None, []
    for line in txt.split("\n"):
        mt = re.search(r"loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)", line)
        if mt:
            last = (float(mt.group(1)), float(mt.group(2)), float(mt.group(3)))
        mv = re.search(r"val_loss:([0-9.]+) valstep:([0-9]+)", line)
        if mv and last is not None:
            out.append((int(mv.group(2)) + offset, last[0], float(mv.group(1)), last[1], last[2]))
    return out


data = parse(TRAIN, 0) + parse(COOL, 30000)
if not data:
    raise SystemExit(f"aucun point parsé pour {RES}")
steps = [d[0] for d in data]; tl = [d[1] for d in data]; vl = [d[2] for d in data]
gr = [d[3] for d in data]; lr = [d[4] for d in data]
vmin = min(range(len(vl)), key=lambda i: vl[i])
CDX = 30000

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
fig.suptitle(f"Apple b2 (1 caméra fixe, batch2 randomisé) — {RES}px, R34+U-Net[128,256,512], "
             f"const 1e-4+EMA+cooldown  ({len(data)} points)", fontsize=12, fontweight="bold")

def cd(a): a.axvline(CDX, color="#94a3b8", ls="--", lw=1); a.axvspan(CDX, max(steps), color="#f1f5f9", alpha=.6, zorder=0)

a = ax[0, 0]
a.plot(steps, tl, "-", color="#2563eb", lw=.8, label="train loss")
a.plot(steps, vl, "-", color="#dc2626", lw=.8, alpha=.85, label="val loss")
cd(a); a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log)")
a.set_title("(A) Loss train vs val (zone grise = cooldown)"); a.legend(); a.grid(alpha=.3)

b = ax[0, 1]
b.plot(steps, vl, "-", color="#dc2626", lw=.8)
b.plot(steps[vmin], vl[vmin], "*", color="#f59e0b", ms=20, markeredgecolor="k",
       label=f"min val = {vl[vmin]:.4f} @ {steps[vmin]}")
cd(b); b.set_xlabel("step"); b.set_ylabel("val loss")
b.set_title("(B) Val-loss — JUGE offline (pas de rollout pomme)"); b.legend(fontsize=9); b.grid(alpha=.3)

c = ax[1, 0]
c.plot(steps, gr, "-", color="#059669", lw=.7)
cd(c); c.set_xlabel("step"); c.set_ylabel("grad norm"); c.set_title("(C) Grad-norm"); c.grid(alpha=.3)

d = ax[1, 1]
d.plot(steps, lr, "-", color="#7c3aed", lw=1.2)
cd(d); d.set_xlabel("step"); d.set_ylabel("lr"); d.set_title("(D) Learning rate (const -> cooldown 0)")
d.grid(alpha=.3); d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=120)
print(f"-> {OUT} | {len(data)} points | min val-loss = {vl[vmin]:.4f} @ {steps[vmin]}")
