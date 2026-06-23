"""Un graphe 2×2 (succès + loss + lr + grad_norm) PAR modèle du tableau de convergence.
Sorties : results/runs/phase5_methodology/full_<nom>.png (embarquées en petit dans CONVERGENCE.md).

Stitch les fragments de logs (loss/grad/lr) et de rollouts d'un même run.
Usage : venv312/bin/python experiments/phase5_methodology/34_plot_full_per_model.py
"""
import csv, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "results/logs/phase5_methodology/"
R = "results/runs/phase5_methodology/"
OUT = R

MODELS = {
    "resnet34_dense": {
        "title": "ResNet34 dense — birdview, batch 32",
        "success": [R + "26_resnet34_dense/rollouts_500.csv"], "rollN": 500,
        "logs": [L + "run_01_train_dense.log"], "color": "#1f77b4"},
    "mini_constant": {
        "title": "mini-CNN constant 1e-4 — batch 32",
        "success": [R + "mini_constant/rollouts_500.csv", R + "mini_constant_continue/rollouts_500.csv",
                    R + "mini_constant_continue2/rollouts_500.csv", R + "mini_constant_resume/rollouts_500.csv",
                    R + "mini_constant_150k/rollouts_500.csv", R + "mini_constant_200k/rollouts_500.csv",
                    R + "mini_constant_250k/rollouts_500.csv"], "rollN": 500,
        "logs": [L + "run_mini_constant.log", L + "run_mini_constant_continue.log",
                 L + "run_mini_constant_continue2.log", L + "run_mini_constant_resume.log",
                 L + "run_mini_constant_150k.log", L + "run_mini_constant_200k.log",
                 L + "run_mini_constant_250k.log"], "color": "#2ca02c"},
    "mini_cosine": {
        "title": "mini-CNN cosine (vagues SGDR) — batch 32",
        "success": [R + "mini_cosine_rollouts_500.csv", R + "mini_cosine_continue/rollouts_500.csv",
                    R + "mini_cosine_continue2/rollouts_500.csv", R + "mini_cosine_150k/rollouts_500.csv",
                    R + "mini_cosine_200k/rollouts_500.csv", R + "mini_cosine_250k/rollouts_500.csv"], "rollN": 500,
        "logs": [L + "run_mini_cosine.log", L + "run_mini_cosine_continue.log",
                 L + "run_mini_cosine_continue2.log", L + "run_mini_cosine_150k.log",
                 L + "run_mini_cosine_200k.log", L + "run_mini_cosine_250k.log"], "color": "#1f77b4"},
    "run31_r34_bigunet": {
        "title": "ResNet34 + gros U-Net (61M) — batch 16",
        "success": ["results/runs/can/31_proprio_birdview_r34_bigunet/rollouts_50.csv"], "rollN": 50,
        "eval500": "results/runs/can/31_proprio_birdview_r34_bigunet/eval500_best.csv",
        "logs": ["results/logs/can/run_31_r34_bigunet.log"], "color": "#d62728"},
}


def smooth(y, w=15):
    y = np.asarray(y, float)
    if len(y) < 3: return y
    h = w // 2
    return np.array([y[max(0, i - h):min(len(y), i + h + 1)].mean() for i in range(len(y))])


def load_logs(logs):
    steps, tl, gr, lr = [], [], [], []
    seen = set()
    for lg in logs:
        try: txt = open(lg, errors="ignore").read()
        except FileNotFoundError: continue
        tr = re.findall(r"(?<!val_)loss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+)", txt)
        vl = re.findall(r"val_loss:[0-9.]+ valstep:(\d+)", txt)
        n = min(len(tr), len(vl))
        for i in range(n):
            s = int(vl[i])
            if s in seen: continue
            seen.add(s); steps.append(s); tl.append(float(tr[i][0])); gr.append(float(tr[i][1])); lr.append(float(tr[i][2]))
    o = np.argsort(steps)
    return np.array(steps)[o], np.array(tl)[o], np.array(gr)[o], np.array(lr)[o]


def load_success(csvs):
    d = {}
    for f in csvs:
        try:
            for r in csv.DictReader(open(f)):
                if r.get("success_rate") not in (None, ""): d[int(float(r["step"]))] = float(r["success_rate"]) * 100
        except FileNotFoundError: pass
    s = sorted(d.items())
    return np.array([k for k, _ in s]), np.array([v for _, v in s])


for name, m in MODELS.items():
    ss, sr = load_success(m["success"])
    st, tl, gr, lr = load_logs(m["logs"])
    if len(ss) == 0 and len(st) == 0:
        print(f"{name}: pas de données, skip"); continue
    c = m["color"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7)); ax = ax.flatten()
    ax[0].plot(ss / 1000, sr, "-o", color=c, ms=4, label=f"{m['rollN']} rollouts")
    if m.get("eval500"):
        e = list(csv.DictReader(open(m["eval500"])))
        ex = np.array([int(float(r["step"])) for r in e]) / 1000
        ey = np.array([float(r["success_rate"]) * 100 for r in e])
        ax[0].plot(ex, ey, "*", color="#2ca02c", ms=16, zorder=5, label="500 rollouts (fiable)")
        ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("Succès (rollouts) %"); ax[0].set_ylim(-3, 102); ax[0].grid(alpha=.3); ax[0].set_xlabel("step (k)")
    if len(st):
        ax[1].plot(st / 1000, smooth(tl), "-", color=c, lw=1.5); ax[1].set_yscale("log")
        ax[1].set_title("loss (lissée) — log"); ax[1].grid(alpha=.3, which="both"); ax[1].set_xlabel("step (k)")
        ax[2].plot(st / 1000, lr, "-", color=c, lw=1.5); ax[2].set_title("learning rate")
        ax[2].grid(alpha=.3); ax[2].set_xlabel("step (k)")
        ax[3].plot(st / 1000, smooth(gr), "-", color=c, lw=1.5); ax[3].set_yscale("log")
        ax[3].set_title("grad_norm — log"); ax[3].grid(alpha=.3, which="both"); ax[3].set_xlabel("step (k)")
    fig.suptitle(m["title"], fontweight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT + f"full_{name}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"{name}: succès {len(ss)} pts, logs {len(st)} pts -> {out}")
