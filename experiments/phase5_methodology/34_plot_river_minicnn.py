#!/usr/bin/env python3
"""Profil rivière mini-CNN Can : CONSTANT 1e-4 vs COSINE SGDR (restart 20/50/80/150/200/250k).
Rebond brut (succès par checkpoint) + fond SWA par fenêtre (barre = étendue), les 2 runs superposés.
Lit les fonds depuis river_{const,cos}_w*/r.csv (auto-remplis par 33_river_minicnn_cos_vs_const.sh).
  venv312/bin/python experiments/phase5_methodology/34_plot_river_minicnn.py
"""
import csv, os, math, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/phase5_methodology"
RESTARTS = [20, 50, 80, 150, 200, 250]  # bornes SGDR (le cosine touche son creux JUSTE AVANT)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n; c = (p + z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-h)*100, min(1.,c+h)*100

def raw(patterns):
    """Agrège le rebond brut depuis TOUS les rollouts_500.csv (racine + sous-dossiers), dedup par step."""
    by_step = {}
    files = []
    for p in patterns: files += glob.glob(p)
    for fp in files:
        for r in csv.DictReader(open(fp)):
            try: s = int(float(r["step"])); v = float(r["success_rate"])*100
            except (KeyError, ValueError): continue
            by_step[s] = v  # dernier vu gagne
    xs = sorted(by_step)
    return [s/1000 for s in xs], [by_step[s] for s in xs]

def floor(tag, label):
    p = f"{R}/river_{tag}_{label}/r.csv"
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    r = rows[-1]; rate = float(r["success_rate"])*100
    lo = float(r.get("ci95_low") or 0)*100; hi = float(r.get("ci95_high") or 0)*100
    n = int(float(r.get("n") or 500))
    if hi <= lo: lo, hi = wilson(round(rate/100*n), n)
    return rate, lo, hi

# (label, start_k, end_k)
WINS = {
    "w1": (4,20), "w2": (25,45), "w3": (60,80), "w4": (85,105),
    "w5": (110,130), "w6": (130,150), "w7": (160,200), "w8": (210,250),
}
CONST_W = ["w1","w2","w3","w4","w5","w6"]
COS_W   = ["w1","w2","w3","w4","w5","w6","w7","w8"]

fig, ax = plt.subplots(figsize=(14, 7))

# rebond brut (agrégé racine + sous-dossiers ; on exclut les dirs SWA river_*/_all/_swa via le motif rollouts_500)
for tag, pats, col, lab in [
    ("const", [f"{R}/mini_constant_rollouts_500.csv", f"{R}/mini_constant/rollouts_500.csv",
               f"{R}/mini_constant_continue*/rollouts_500.csv", f"{R}/mini_constant_resume/rollouts_500.csv",
               f"{R}/mini_constant_1*0k/rollouts_500.csv", f"{R}/mini_constant_2*0k/rollouts_500.csv"],
     "#e8a0a0", "constant — brut/checkpoint (n=500)"),
    ("cos",   [f"{R}/mini_cosine_rollouts_500.csv", f"{R}/mini_cosine/rollouts_500.csv",
               f"{R}/mini_cosine_continue*/rollouts_500.csv",
               f"{R}/mini_cosine_1*0k/rollouts_500.csv", f"{R}/mini_cosine_2*0k/rollouts_500.csv",
               f"{R}/mini_cosine_3*0k/rollouts_500.csv"],
     "#9db8d8", "cosine SGDR — brut/checkpoint (n=500)")]:
    xs, ys = raw(pats)
    if xs: ax.plot(xs, ys, "-o", color=col, ms=2.5, lw=0.9, alpha=0.8, label=lab)

# fonds SWA
for tag, wlist, col, lab in [
    ("const", CONST_W, "#c0392b", "fond SWA — CONSTANT (fenêtre 5 ckpts)"),
    ("cos",   COS_W,   "#1f4e8c", "fond SWA — COSINE SGDR (fenêtre 5 ckpts)")]:
    for w in wlist:
        res = floor(tag, w)
        if res is None: continue
        f, lo, hi = res; a, b = WINS[w]; mid = (a+b)/2
        ax.hlines(f, a, b, color=col, lw=5, alpha=0.9, zorder=5)
        ax.plot([a, b], [f, f], "|", color=col, ms=10, mew=2, zorder=6)
        ax.errorbar(mid, f, yerr=[[f-lo],[hi-f]], color=col, capsize=4, elinewidth=1.3, alpha=0.6, zorder=6)
        dy = 5 if tag == "cos" else -10
        ax.annotate(f"{f:.0f}", (mid, f), color=col, fontsize=8.5, fontweight="bold",
                    xytext=(0, dy), textcoords="offset points", ha="center", zorder=7)

# restarts SGDR
for x in RESTARTS:
    ax.axvline(x, color="#888", ls=":", lw=1.1, zorder=1)
    ax.annotate(f"restart\n{x}k", (x, 2), color="#777", fontsize=7.5, ha="center", va="bottom")

ax.set_xlabel("step d'entraînement (k)", fontsize=12)
ax.set_ylabel("succès (rollouts Can) %", fontsize=12)
ax.set_ylim(-3, 100); ax.set_xlim(0, 258); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
ax.set_title("mini-CNN Can — profil « rivière » : CONSTANT 1e-4 vs COSINE SGDR (restart /vague)\n"
             "barre = fond SWA d'une fenêtre (5 ckpts) ; lignes claires = rebond brut ; pointillés = restarts SGDR",
             fontsize=11.5)
fig.tight_layout(); out = f"{R}/courbes_river_minicnn.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, "| fonds :",
      ", ".join(f"{t}:{w}={floor(t,w)[0]:.0f}" for t,wl in [("const",CONST_W),("cos",COS_W)]
               for w in wl if floor(t,w) is not None))
