#!/usr/bin/env python
"""4-panneaux SANS val (modèle cartésien combiné entraîné sur tout) : train-loss(log) / train-loss+min / grad / lr."""
import re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN = Path("results/logs/real/apple_cart_combined.log")
COOL = Path("results/logs/real/cd_cart_combined.log")
OUT = Path("results/runs/real/apple_cart_combined_128/cart_combined_4panel.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
CDX = 40000


def parse(p, offset=0):
    if not p.exists():
        return {}
    txt = p.read_text(errors="ignore").replace("\r", "\n")
    d = {}
    # le vrai step = compteur tqdm "N/Total" ; loss/grad/lr sur la même ligne (step:XXK est abrégé, inutilisable)
    for line in txt.split("\n"):
        m = re.search(r"(\d+)/\d+ \[.*?loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.e+-]+)", line)
        if m:
            d[int(m.group(1)) + offset] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return d


d = parse(TRAIN, 0); d.update(parse(COOL, CDX))
steps = sorted(d)
loss = [d[s][0] for s in steps]; gr = [d[s][1] for s in steps]; lr = [d[s][2] for s in steps]
lmin = min(range(len(loss)), key=lambda i: loss[i])

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
fig.suptitle(f"Apple CARTÉSIEN COMBINÉ (classique + corrections) — 263M, R18+U-Net[512,1024,2048]+crop112, "
             f"128px 15Hz, const 1e-4+EMA+cooldown  ({len(steps)} points, PAS de val)", fontsize=11, fontweight="bold")


def cd(a):
    a.axvline(CDX, color="#94a3b8", ls="--", lw=1); a.axvspan(CDX, max(steps), color="#f1f5f9", alpha=.6, zorder=0)


a = ax[0, 0]
a.plot(steps, loss, "-", color="#2563eb", lw=.7)
cd(a); a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("train loss (log)")
a.set_title("(A) Train loss — échelle log (zone grise = cooldown)"); a.grid(alpha=.3)

b = ax[0, 1]
b.plot(steps, loss, "-", color="#2563eb", lw=.7)
b.plot(steps[lmin], loss[lmin], "*", color="#f59e0b", ms=20, markeredgecolor="k",
       label=f"min loss = {loss[lmin]:.4f} @ {steps[lmin]}")
cd(b); b.set_xlabel("step"); b.set_ylabel("train loss")
b.set_title("(B) Train loss + min (⚠️ pas de val — juge = robot)"); b.legend(fontsize=9); b.grid(alpha=.3)

c = ax[1, 0]
c.plot(steps, gr, "-", color="#059669", lw=.6)
cd(c); c.set_xlabel("step"); c.set_ylabel("grad norm"); c.set_title("(C) Grad-norm"); c.grid(alpha=.3)

e = ax[1, 1]
e.plot(steps, lr, "-", color="#7c3aed", lw=1.2)
cd(e); e.set_xlabel("step"); e.set_ylabel("lr"); e.set_title("(D) Learning rate (const → cooldown 0)")
e.grid(alpha=.3); e.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=120)
print(f"-> {OUT} | {len(steps)} points | min train-loss = {loss[lmin]:.4f} @ {steps[lmin]}")
