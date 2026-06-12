"""Courbe COMPLETE mini-CNN 1k->50k : cosine (vague SGDR) vs LR constant.

Fusionne la phase initiale (1k->20k) et la phase continue (21k->50k) sur 4 panneaux :
  succes 500-rollouts (+IC95) | loss train(—)+val(- -) | learning rate (lineaire, sci) | grad_norm.
Ligne verticale a 20k = jonction (reprise + warm restart cosine).

Usage : venv312/bin/python experiments/phase5_methodology/15_plot_minicnn_full.py
"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
LOGS = Path("results/logs/phase5_methodology")
# log : (chemin, offset_step) — la phase continue logge sa barre tqdm de 0..30000, on la decale de +20000
RUNS = {
    "cosine (vague SGDR)": {
        "csv":  [ROOT / "mini_cosine_rollouts_500.csv", ROOT / "mini_cosine_continue" / "rollouts_500.csv"],
        "log":  [(LOGS / "run_mini_cosine.log", 0),     (LOGS / "run_mini_cosine_continue.log", 20000)],
        "c": "tab:blue"},
    "constant 1e-4": {
        "csv":  [ROOT / "mini_constant" / "rollouts_500.csv", ROOT / "mini_constant_continue" / "rollouts_500.csv"],
        "log":  [(LOGS / "run_mini_constant.log", 0),         (LOGS / "run_mini_constant_continue.log", 20000)],
        "c": "tab:red"},
}

def read_success(paths):
    steps, sr, lo, hi = [], [], [], []
    for p in paths:
        for r in csv.DictReader(open(p)):
            steps.append(int(r["step"])); sr.append(float(r["success_rate"]) * 100)
            lo.append(float(r["ci95_low"]) * 100); hi.append(float(r["ci95_high"]) * 100)
    z = sorted(zip(steps, sr, lo, hi))
    return [list(t) for t in zip(*z)]

def read_train(specs):
    s_l, loss, grdn, lr = [], [], [], []
    for p, off in specs:
        txt = Path(p).read_text(errors="ignore").replace("\r", "\n")
        for line in txt.split("\n"):
            m = re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
            if m:
                s_l.append(int(m.group(1)) + off); loss.append(float(m.group(2)))
                grdn.append(float(m.group(3))); lr.append(float(m.group(4)))
    z = sorted(zip(s_l, loss, grdn, lr))
    return [list(t) for t in zip(*z)]

def read_valloss(specs):
    steps, vl = [], []
    for p, _off in specs:   # valstep est deja le vrai step (20000+ dans la phase continue)
        txt = Path(p).read_text(errors="ignore").replace("\r", "\n")
        for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", txt):
            vl.append(float(m.group(1))); steps.append(int(m.group(2)))
    z = sorted(zip(steps, vl))
    return [list(t) for t in zip(*z)]

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("mini-CNN sur Can (vision pure, 1.84M params) — 1k→50k : cosine (vague) vs LR constant",
             fontsize=13, weight="bold")

for name, d in RUNS.items():
    col = d["c"]
    st, sr, lo, hi = read_success(d["csv"])
    ax[0, 0].plot(st, sr, "-o", color=col, ms=3.5, label=name)
    ax[0, 0].fill_between(st, lo, hi, color=col, alpha=0.15)
    tst, tl, gn, lr = read_train(d["log"])
    vst, vl = read_valloss(d["log"])
    ax[0, 1].plot(tst, tl, "-",  color=col, alpha=0.5, lw=0.9, label=f"{name} train")
    ax[0, 1].plot(vst, vl, "--", color=col, alpha=0.95, lw=1.2, label=f"{name} val")
    ax[1, 0].plot(tst, lr, "-", color=col, lw=1.5, label=name)
    ax[1, 1].plot(tst, gn, "-", color=col, alpha=0.8, lw=1, label=name)
    print(f"{name:20s}: succes max {max(sr):.1f}% @ step {st[sr.index(max(sr))]}  (fin 50k: {sr[-1]:.1f}%)")

for a in ax.flat:
    a.axvline(20000, ls=":", c="gray", lw=1)          # jonction reprise
ax[0, 0].set_title("Succès 500-rollouts (+ IC95 Wilson)"); ax[0, 0].set_ylabel("succès (%)")
ax[0, 0].axhline(2, ls=":", c="lightgray", lw=0.8)
ax[0, 1].set_title("loss train (—) + val (- -) — échelle log"); ax[0, 1].set_ylabel("loss"); ax[0, 1].set_yscale("log")
ax[1, 0].set_title("learning rate"); ax[1, 0].set_ylabel("LR")
ax[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax[1, 1].set_title("grad_norm — échelle log"); ax[1, 1].set_ylabel("grad_norm"); ax[1, 1].set_yscale("log")
for a in ax.flat:
    a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)

fig.tight_layout()
out = ROOT / "courbes_minicnn_full_1k_50k.png"
fig.savefig(out, dpi=150)
print(f"écrit : {out}")
