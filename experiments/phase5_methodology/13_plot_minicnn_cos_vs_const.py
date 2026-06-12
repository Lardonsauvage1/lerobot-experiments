"""Courbe comparative mini-CNN : cosine vs constant (LR), runs 0->20k.

4 panneaux : succes 500-rollouts (+IC95) | val_loss | learning rate (log) | grad_norm.
Succes : *_rollouts_500.csv ; loss/lr/grdn : extraits des logs d'entrainement.

Usage : venv312/bin/python experiments/phase5_methodology/13_plot_minicnn_cos_vs_const.py
"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
LOGS = Path("results/logs/phase5_methodology")
RUNS = {
    "cosine":   {"csv": ROOT / "mini_cosine_rollouts_500.csv",          "log": LOGS / "run_mini_cosine.log",   "c": "tab:blue"},
    "constant": {"csv": ROOT / "mini_constant" / "rollouts_500.csv",    "log": LOGS / "run_mini_constant.log", "c": "tab:red"},
}

def read_success(p):
    steps, sr, lo, hi = [], [], [], []
    for r in csv.DictReader(open(p)):
        steps.append(int(r["step"])); sr.append(float(r["success_rate"]) * 100)
        lo.append(float(r["ci95_low"]) * 100); hi.append(float(r["ci95_high"]) * 100)
    return steps, sr, lo, hi

def read_train(p):
    """Renvoie (steps, loss, grdn, lr) depuis les lignes tqdm + INFO step:.. loss:.. grdn:.. lr:.."""
    txt = Path(p).read_text(errors="ignore").replace("\r", "\n")
    s_l, loss, grdn, lr = [], [], [], []
    for line in txt.split("\n"):
        m = re.search(r"\|\s*(\d+)/\d+.*loss:([\d.]+) grdn:([\d.]+) lr:([\d.eE+-]+)", line)
        if m:
            s_l.append(int(m.group(1))); loss.append(float(m.group(2)))
            grdn.append(float(m.group(3))); lr.append(float(m.group(4)))
    return s_l, loss, grdn, lr

def read_valloss(p):
    txt = Path(p).read_text(errors="ignore").replace("\r", "\n")
    steps, vl = [], []
    for m in re.finditer(r"val_loss:([\d.]+) valstep:(\d+)", txt):
        vl.append(float(m.group(1))); steps.append(int(m.group(2)))
    return steps, vl

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("mini-CNN sur Can (vision pure, 1.84M params) — cosine vs LR constant, 0→20k", fontsize=13, weight="bold")

for name, d in RUNS.items():
    col = d["c"]
    st, sr, lo, hi = read_success(d["csv"])
    ax[0, 0].plot(st, sr, "-o", color=col, ms=4, label=name)
    ax[0, 0].fill_between(st, lo, hi, color=col, alpha=0.15)
    tst, tl, gn, lr = read_train(d["log"])
    vst, vl = read_valloss(d["log"])
    ax[0, 1].plot(tst, tl, "-",  color=col, alpha=0.5, lw=0.9, label=f"{name} train")
    ax[0, 1].plot(vst, vl, "--", color=col, alpha=0.95, lw=1.3, label=f"{name} val")
    ax[1, 0].plot(tst, lr, "-", color=col, lw=1.5, label=name)
    ax[1, 1].plot(tst, gn, "-", color=col, alpha=0.8, lw=1, label=name)

ax[0, 0].set_title("Succès 500-rollouts (+ IC95 Wilson)"); ax[0, 0].set_ylabel("succès (%)")
ax[0, 0].set_ylim(-0.5, 12); ax[0, 0].axhline(2, ls=":", c="gray", lw=0.8)
ax[0, 1].set_title("loss train (—) + val (- -)"); ax[0, 1].set_ylabel("loss")
ax[1, 0].set_title("learning rate"); ax[1, 0].set_ylabel("LR")
ax[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax[1, 1].set_title("grad_norm"); ax[1, 1].set_ylabel("grad_norm")
for a in ax.flat:
    a.set_xlabel("step"); a.legend(); a.grid(alpha=0.3)

fig.tight_layout()
out = ROOT / "courbes_minicnn_cos_vs_const.png"
fig.savefig(out, dpi=150)
print(f"écrit : {out}")
# petit récap console
for name, d in RUNS.items():
    st, sr, _, _ = read_success(d["csv"])
    print(f"{name:9s}: {len(st)} pts, succès max {max(sr):.1f}% (step {st[sr.index(max(sr))]})")
