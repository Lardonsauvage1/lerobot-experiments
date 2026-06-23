"""Étude de convergence : succès (rollouts) vs steps d'entraînement.

AUTO-EXTENSIBLE : découvre seul tous les `**/*rollouts_500.csv` et `**/*rollouts_50.csv`
sous results/runs/, recolle les fragments d'un même run (_continue/_resume/_150k/...),
exclut les séries camshift (camaug = autre mesure).

Mesure : 500 rollouts (IC95 ±~4 pts) = référence. 50 rollouts (IC95 ±~13 pts) =
provisoire, tracé en pointillés. Pour un même run, le 500r prime sur le 50r.

=> Ajouter un modèle : produire un `<run>/rollouts_500.csv` (ou `rollouts_50.csv`)
   à checkpoints réguliers, puis relancer ce script.

Sorties : results/runs/phase5_methodology/convergence_rollouts.png + convergence_metrics.md
"""
import csv
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/runs/"
OUT_PNG = "results/runs/phase5_methodology/convergence_rollouts.png"
OUT_MD = "results/runs/phase5_methodology/convergence_metrics.md"

LABELS = {
    "26_resnet34_dense": "ResNet34 dense (gros modèle)",
    "31_proprio_birdview_r34_bigunet": "ResNet34 + gros U-Net (61M)",
    "mini_cosine": "mini-CNN cosine (minuscule)",
    "mini_constant": "mini-CNN constant (minuscule)",
}


def family(path):
    parts = path.split("/")
    base = parts[-1]
    if base in ("rollouts_500.csv", "rollouts_50.csv") and parts[-2] != "phase5_methodology":
        name = parts[-2]
    else:
        name = re.sub(r"_rollouts_(500|50)$", "", base.replace(".csv", ""))
    while True:
        new = re.sub(r"(_continue\d*|_resume|_finevar|_ext|_\d+k)$", "", name)
        if new == name:
            break
        name = new
    return name


def load(f):
    out = []
    for r in csv.DictReader(open(f)):
        if r.get("step") and r.get("success_rate") not in (None, ""):
            out.append((int(float(r["step"])), float(r["success_rate"])))
    return out


# --- découverte : par famille, on stocke séparément 500r et 50r ---
raw = {}  # fam -> {500:{step:sr}, 50:{step:sr}}
for N, pat in [(500, "*rollouts_500.csv"), (50, "*rollouts_50.csv")]:
    for f in sorted(glob.glob(ROOT + "**/" + pat, recursive=True)):
        if "camaug" in f.lower() or "camshift" in f.lower():
            continue
        fam = family(f)
        raw.setdefault(fam, {500: {}, 50: {}})
        for st, sr in load(f):
            raw[fam][N].setdefault(st, sr)

# choix par famille : 500r prime, sinon 50r (provisoire)
curves = {}  # fam -> (points triés, N)
for fam, d in raw.items():
    if len(d[500]) >= 3:
        curves[fam] = (sorted(d[500].items()), 500)
    elif len(d[50]) >= 3:
        curves[fam] = (sorted(d[50].items()), 50)


def metrics(c):
    st = np.array([x[0] for x in c]); sr = np.array([x[1] for x in c])
    mx = sr.max()
    def reach(frac):
        m = sr >= frac * mx
        return int(st[np.argmax(m)]) if m.any() else None
    decoll = int(st[np.argmax(sr > 0.05)]) if (sr > 0.05).any() else None
    return dict(n=len(c), smin=int(st.min()), smax=int(st.max()), mx=mx,
                decoll=decoll, t50=reach(0.5), t90=reach(0.9))


order = sorted(curves.items(), key=lambda kv: -metrics(kv[1][0])["mx"])

# --- tableau ---
lines = ["| run | rollouts | n pts | plage steps | plafond | décollage (>5%) | 50% du max | 90% du max |",
         "|---|---|---|---|---|---|---|---|"]
for fam, (c, N) in order:
    m = metrics(c); lbl = LABELS.get(fam, fam)
    tag = "500" if N == 500 else "50 (prov.)"
    lines.append(f"| {lbl} | {tag} | {m['n']} | {m['smin']}–{m['smax']} | {m['mx']:.0%} | "
                 f"{m['decoll']} | {m['t50']} | {m['t90']} |")
md = ("# Convergence — métriques (succès en rollouts vs steps)\n\n"
      "*Généré par `experiments/phase5_methodology/33_plot_convergence_rollouts.py`.*\n\n"
      + "\n".join(lines) + "\n")
open(OUT_MD, "w").write(md)
print(md)

# --- graphe ---
cmap = plt.get_cmap("tab10")
fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2, 1]})
for i, (fam, (c, N)) in enumerate(order):
    st = np.array([x[0] for x in c]) / 1000; sr = np.array([x[1] for x in c]) * 100
    col = cmap(i % 10); lbl = LABELS.get(fam, fam) + (" — 50r prov." if N == 50 else "")
    ls = "--" if N == 50 else "-"
    ax.plot(st, sr, ls, marker="o", ms=2.5, lw=1.5, color=col, label=lbl, alpha=0.9 if N == 500 else 0.7)
    axz.plot(st, sr, ls, marker="o", ms=3.5, lw=1.8, color=col, alpha=0.9 if N == 500 else 0.7)
ax.set_xlabel("steps d'entraînement (k)"); ax.set_ylabel("succès (rollouts) %")
ax.set_title("Convergence du succès vs steps — effet de la capacité")
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8); ax.set_ylim(-3, 102)
axz.set_xlim(0, 40); axz.set_ylim(-3, 102); axz.grid(alpha=0.3)
axz.set_xlabel("steps (k) — zoom 0-40k"); axz.set_title("Zoom : décollage")
fig.suptitle("Étude de convergence (rollouts réguliers) — Can", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT_PNG, dpi=160)
print("Sauvé :", OUT_PNG, "+", OUT_MD)
