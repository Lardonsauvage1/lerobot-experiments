#!/usr/bin/env python3
"""mini-CNN CONSTANT — profil COOLDOWN (LR=0) vs MERGE (SWA) vs BRUT, grille 5k 0-100k.
Lit cooldown_profile_minicnn.csv / merge_profile_minicnn.csv (+ rebond brut des rollouts_500).
  venv312/bin/python experiments/phase5_methodology/92_plot_minicnn_cooldown.py
"""
import csv, os, glob, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/can"; P5 = "results/runs/phase5_methodology"

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-h)*100, min(1.,c+h)*100

def prof(path):
    xs, ys, lo, hi = [], [], [], []
    if not os.path.exists(path): return xs, ys, lo, hi
    for r in csv.DictReader(open(path)):
        if r.get("success_rate") in ("", None): continue   # éval ratée -> ignorer
        k=int(float(r["source_k"])); rate=float(r["success_rate"])*100
        a=float(r.get("ci95_low") or 0)*100; b=float(r.get("ci95_high") or 0)*100
        n=int(float(r.get("n") or 500))
        if b<=a: a,b=wilson(round(rate/100*n),n)
        xs.append(k); ys.append(rate); lo.append(rate-a); hi.append(b-rate)
    z=sorted(zip(xs,ys,lo,hi)); return [list(t) for t in zip(*z)] if z else ([],[],[],[])

# brut constant (agrégé)
raw = {}
for pat in [f"{P5}/mini_constant_rollouts_500.csv", f"{P5}/mini_constant/rollouts_500.csv",
            f"{P5}/mini_constant_continue*/rollouts_500.csv", f"{P5}/mini_constant_resume/rollouts_500.csv",
            f"{P5}/mini_constant_1*0k/rollouts_500.csv"]:
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            try: raw[int(float(r["step"]))]=float(r["success_rate"])*100
            except: pass
GRID = list(range(5, 101, 5))
rx=[k for k in GRID if k*1000 in raw]; ry=[raw[k*1000] for k in rx]

# brut COSINE (agrégé) — pour comparer constant vs cosine sur le même graphe
rawc = {}
for pat in [f"{P5}/mini_cosine_rollouts_500.csv", f"{P5}/mini_cosine/rollouts_500.csv",
            f"{P5}/mini_cosine_continue*/rollouts_500.csv", f"{P5}/mini_cosine_1*0k/rollouts_500.csv"]:
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            try: rawc[int(float(r["step"]))]=float(r["success_rate"])*100
            except: pass
cxr=sorted(k for k in rawc if k<=100000); cyr=[rawc[k] for k in cxr]

cx, cy, clo, chi = prof(f"{R}/cooldown_profile_minicnn.csv")
mx, my, _, _ = prof(f"{R}/merge_profile_minicnn.csv")

fig, ax = plt.subplots(figsize=(12, 6.5))
if cxr: ax.plot([k/1000 for k in cxr], cyr, "-", color="#7fb0e0", lw=1.4, alpha=0.85, label="brut COSINE SGDR (1 ckpt, n=500)")
if rx: ax.plot(rx, ry, "--^", color="#b0b0b0", ms=6, lw=1.3, alpha=0.9, label="brut constant (1 ckpt, n=500)")
if mx: ax.plot(mx, my, "-s", color="#d62728", ms=6, lw=1.7, label="merge SWA — 5 ckpts centrés")
if cx:
    ax.errorbar(cx, cy, yerr=[clo, chi], color="#1f2d3d", marker="o", ms=8, lw=2.3, capsize=4,
                zorder=6, label="cooldown LR=0 (annealing, n=500)")
    for x, y in zip(cx, cy): ax.annotate(f"{y:.0f}", (x, y), color="#1f2d3d", fontsize=8,
                                          xytext=(0, 8), textcoords="offset points", ha="center")
ax.axhline(80, color="#27ae60", ls="--", lw=1.2, alpha=0.6, label="plateau cosine SGDR ~80 %")
ax.set_xlabel("checkpoint de départ (k steps)", fontsize=12)
ax.set_ylabel("succès (rollouts Can) %", fontsize=12)
ax.set_ylim(0, 100); ax.set_xlim(0, 102); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
ax.set_title("mini-CNN Can CONSTANT — COOLDOWN (LR→0) vs MERGE vs BRUT, grille 5k (0-100k)\n"
             "l'annealing rejoint-il / dépasse-t-il le merge ? le brut rebondit, le cooldown se pose", fontsize=11)
fig.tight_layout(); out=f"{P5}/courbes_minicnn_cooldown_profile.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, "| cooldown :", ", ".join(f"{x}={y:.0f}" for x,y in zip(cx,cy)) if cx else "(vide)")
