"""Courbe robustesse au deplacement camera : succes vs niveau, 2 checkpoints (74% vs 56%)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")
RUNS = {"ckpt 46000 (baseline 74%)": ("camshift_46k.csv", "tab:red"),
        "ckpt 45000 (baseline 56%, voisin)": ("camshift_45k.csv", "tab:blue")}

fig, ax = plt.subplots(figsize=(10, 6))
for name, (fn, col) in RUNS.items():
    x, sr, lo, hi = [], [], [], []
    for r in csv.DictReader(open(ROOT / fn)):
        x.append(float(r["trans_cm"])); sr.append(float(r["success_rate"])*100)
        lo.append(float(r["ci95_low"])*100); hi.append(float(r["ci95_high"])*100)
    ax.plot(x, sr, "-o", color=col, ms=6, label=name)
    ax.fill_between(x, lo, hi, color=col, alpha=0.15)
    for xi, s in zip(x, sr): ax.annotate(f"{s:.0f}%", (xi, s), textcoords="offset points", xytext=(6, 6), fontsize=8, color=col)

ax.set_title("Robustesse au DÉPLACEMENT des caméras (2 cams jittées aléatoirement/épisode)\n"
             "modèle Can 2-cams — succès 500-rollouts vs ampleur du décalage", fontsize=12, weight="bold")
ax.set_xlabel("décalage caméra par épisode (translation cm / rotation °, aléatoire ≤ niveau)")
ax.set_ylabel("succès (%)"); ax.set_ylim(-2, 80)
ax.set_xticks([0, 2, 5, 10]); ax.set_xticklabels(["0\n(baseline)", "≤2cm / ≤2°", "≤5cm / ≤5°", "≤10cm / ≤10°"])
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "courbe_camshift.png"; fig.savefig(out, dpi=150)
print(f"écrit : {out}")
