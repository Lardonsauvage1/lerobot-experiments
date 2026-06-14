"""Courbe succes 1k -> N (auto-decouverte des tranches) : constant 1e-4 vs cosine (vagues SGDR).
Se met a jour seule au fur et a mesure que la chaine produit mini_*_<N>k/rollouts_500.csv.
"""
import csv, glob, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/phase5_methodology")

def merge_glob(patterns, finevar_only_57k=False):
    d = {}
    files = []
    for pat in patterns:
        files += glob.glob(str(ROOT / pat))
    for fp in files:
        is_fv = "finevar" in fp
        for r in csv.DictReader(open(fp)):
            s = int(r["step"])
            if is_fv and finevar_only_57k and s != 57000:
                continue
            d[s] = (float(r["success_rate"])*100, float(r["ci95_low"])*100, float(r["ci95_high"])*100)
    st = sorted(d)
    return st, [d[s][0] for s in st], [d[s][1] for s in st], [d[s][2] for s in st]

CONST = merge_glob(["mini_constant/rollouts_500.csv", "mini_constant_*/rollouts_500.csv"], finevar_only_57k=True)
COS   = merge_glob(["mini_cosine_rollouts_500.csv", "mini_cosine_*/rollouts_500.csv"])

fig, ax = plt.subplots(figsize=(15, 6))
for (name, col, data) in [("constant 1e-4", "tab:red", CONST), ("cosine (vagues SGDR)", "tab:blue", COS)]:
    st, sr, lo, hi = data
    ax.plot(st, sr, "-o", color=col, ms=2.5, lw=1, label=f"{name} (max {max(sr):.0f}%, fin {sr[-1]:.0f}%)")
    ax.fill_between(st, lo, hi, color=col, alpha=0.12)
maxstep = max(CONST[0][-1], COS[0][-1])
for x in range(50000, maxstep+1, 50000):
    ax.axvline(x, ls=":", c="gray", lw=0.6)
ax.set_title(f"mini-CNN Can — succes 500-rollouts 1k->{maxstep//1000}k : LR constant vs cosine (restart vague /50k)",
             fontsize=12, weight="bold")
ax.set_xlabel("step"); ax.set_ylabel("succes (%)"); ax.set_ylim(-2, 88)
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "courbe_progressive.png"; fig.savefig(out, dpi=150)
print(f"ecrit : {out}  (constant -> {CONST[0][-1]}, cosine -> {COS[0][-1]})")
