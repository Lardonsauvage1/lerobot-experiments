"""Vue complète d'un run, mise en forme façon courbes_minicnn_valfull (2×2) :
  - Succès 50-rollouts (+ IC95)
  - val_loss (lisse) vs train live (○) — log
  - learning rate
  - grad_norm — log
Montre le découplage net loss↔succès (loss converge tôt, succès tard).

Usage : venv312/bin/python experiments/can/35_plot_run31_full.py
"""
import csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "results/runs/can/31_proprio_birdview_r34_bigunet"
LOG = "results/logs/can/run_31_r34_bigunet.log"
OUT = RUN + "/full_curves.png"
C = "#d62728"        # couleur run 31
CL = "#f4a3a3"       # variante claire (train live)

txt = open(LOG, errors="ignore").read()
tr = re.findall(r"(?<!val_)loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)", txt)
vl = re.findall(r"val_loss:([0-9.]+) valstep:(\d+)", txt)
steps = np.array([int(s) for _, s in vl])
n = min(len(tr), len(steps))
steps = steps[:n]
val = np.array([float(v) for v, _ in vl])[:n]
tloss = np.array([float(l) for l, _, _ in tr])[:n]
grad = np.array([float(g) for _, g, _ in tr])[:n]
lr = np.array([float(x) for _, _, x in tr])[:n]


def smooth(y, w=15):
    """Moyenne glissante centrée, fenêtre rétrécie aux bords (PAS de zéro-padding,
    sinon la courbe lissée plonge artificiellement vers 0 aux extrémités)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    half = w // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - half):min(len(y), i + half + 1)].mean()
    return out


# succès dense (+ IC95)
sc = list(csv.DictReader(open(RUN + "/rollouts_50.csv")))
ss = np.array([int(float(r["step"])) for r in sc])
sr = np.array([float(r["success_rate"]) * 100 for r in sc])
lo = np.array([float(r["ci95_low"]) * 100 for r in sc])
hi = np.array([float(r["ci95_high"]) * 100 for r in sc])

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(2, 2, figsize=(16, 10))

# --- succès + IC95 (trajectoire 50 rollouts) ---
a = ax[0, 0]
a.fill_between(ss, lo, hi, color=C, alpha=0.18)
a.plot(ss, sr, "-o", color=C, ms=5, label="50 rollouts (trajectoire, ±13 pts)")
# point(s) fiable(s) à 500 rollouts en surimpression
import os
if os.path.exists(RUN + "/eval500_best.csv"):
    e5 = list(csv.DictReader(open(RUN + "/eval500_best.csv")))
    e5s = np.array([int(float(r["step"])) for r in e5])
    e5r = np.array([float(r["success_rate"]) * 100 for r in e5])
    e5lo = np.array([float(r["ci95_low"]) * 100 for r in e5])
    e5hi = np.array([float(r["ci95_high"]) * 100 for r in e5])
    a.errorbar(e5s, e5r, yerr=[e5r - e5lo, e5hi - e5r], fmt="*", color="#2ca02c",
               ms=18, capsize=4, lw=1.5, zorder=5, label="500 rollouts (fiable, ±4 pts)")
    for x, y in zip(e5s, e5r):
        a.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(-44, -2),
                   fontsize=10, color="#2ca02c", fontweight="bold")
a.set_title("Succès vs steps — trajectoire 50r + point fiable 500r")
a.set_xlabel("step"); a.set_ylabel("succès (%)")
a.set_ylim(-3, 102); a.grid(alpha=0.3); a.legend(loc="lower right", fontsize=9)

# --- val_loss lisse vs train live (log) ---
a = ax[0, 1]
ds = slice(None, None, 5)  # sous-échantillonne les markers train (800 pts -> ~160)
a.plot(steps[ds], tloss[ds], "o", color=CL, ms=3, alpha=0.6, label="train (live)")
a.plot(steps, smooth(val), "-", color=C, lw=2, label="val_loss (lisse)")
a.set_yscale("log"); a.set_title("val_loss (lisse) vs train live (○) — log")
a.set_xlabel("step"); a.set_ylabel("loss"); a.grid(alpha=0.3, which="both"); a.legend(loc="upper right")

# --- learning rate ---
a = ax[1, 0]
a.plot(steps, lr, "-o", color=C, ms=3)
a.set_title("learning rate"); a.set_xlabel("step"); a.set_ylabel("LR"); a.grid(alpha=0.3)

# --- grad_norm (log) ---
a = ax[1, 1]
a.plot(steps, grad, "-", color=C, lw=0.9, alpha=0.5)
a.plot(steps, smooth(grad), "-", color=C, lw=2)
a.set_yscale("log"); a.set_title("grad_norm — log")
a.set_xlabel("step"); a.set_ylabel("grad_norm"); a.grid(alpha=0.3, which="both")

fig.suptitle("Run 31 — ResNet34 + gros U-Net (61M, batch 16) : découplage net loss↔succès",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print("Sauvé :", OUT)
print(f"loss/grad/lr: {n} pts (->{steps[-1]}) ; succès: {len(sr)} ckpts (->{ss[-1]}, max {sr.max():.0f}%)")
