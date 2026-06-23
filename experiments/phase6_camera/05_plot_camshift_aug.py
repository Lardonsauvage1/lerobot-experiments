"""Camshift : baseline non-augmentee vs modeles entraines AVEC augmentation camera (150 / 30 demos).
L'augmentation (1 jitter/demo) rend-elle robuste ? + grad_norm des 2 runs (confirme phase precedente).
"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase6_camera")
LOGS = Path("results/logs/phase6_camera")

def cs(fn):  # dedup par trans_cm (garde le dernier)
    d = {}
    for r in csv.DictReader(open(ROOT/fn)):
        d[float(r["trans_cm"])] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
    x = sorted(d); return x, [d[t][0] for t in x], [d[t][1] for t in x], [d[t][2] for t in x]

CURVES = [("baseline NON-augmentée (74%)", "tab:gray", "camshift_46k.csv"),
          ("augmenté 150 démos", "tab:green", "camshift_camaug150.csv"),
          ("augmenté 30 démos", "tab:orange", "camshift_camaug30.csv")]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for name, col, fn in CURVES:
    if not (ROOT/fn).exists(): continue
    x, sr, lo, hi = cs(fn)
    ax.plot(x, sr, "-o", color=col, ms=6, label=name)
    ax.fill_between(x, lo, hi, color=col, alpha=0.12)
    for xi, s in zip(x, sr): ax.annotate(f"{s:.0f}", (xi, s), textcoords="offset points", xytext=(5, 5), fontsize=7, color=col)
ax.set_title("Robustesse au déplacement caméra : augmentation aide-t-elle ?", fontsize=12, weight="bold")
ax.set_xlabel("décalage caméra (cm translation / ° rotation, aléatoire/épisode)")
ax.set_ylabel("succès (%)"); ax.set_ylim(-2, 80)
ax.set_xticks([0, 2, 5, 10]); ax.set_xticklabels(["0\nnominal", "2cm/2°", "5cm/5°", "10cm/10°"])
ax.legend(); ax.grid(alpha=0.3)

# grad_norm des 2 runs camaug
def grad(fn):
    s, g = [], []
    for line in (LOGS/fn).read_text(errors="ignore").replace("\r","\n").split("\n"):
        m = re.search(r"\|\s*(\d+)/\d+.*grdn:([\d.]+)", line)
        if m: s.append(int(m.group(1))); g.append(float(m.group(2)))
    z = sorted(zip(s, g)); return [list(t) for t in zip(*z)] if z else ([], [])
for fn, name, col in [("run_camaug_150.log","camaug 150 démos","tab:green"),("run_camaug_30.log","camaug 30 démos","tab:orange")]:
    if (LOGS/fn).exists():
        gs, gg = grad(fn)
        if gs: ax2.plot(gs, gg, "-", color=col, lw=1, label=name)
ax2.set_title("grad_norm (cosine 70k) — confirme « le gradient dit quand, pas le niveau »"); ax2.set_xlabel("step")
ax2.set_ylabel("grad_norm"); ax2.set_yscale("log"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.tight_layout(); out = ROOT/"courbe_camshift_aug.png"; fig.savefig(out, dpi=150)
print(f"écrit : {out}")
