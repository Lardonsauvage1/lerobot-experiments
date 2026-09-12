#!/usr/bin/env python3
"""Profil COOLDOWN (checkpoint LR=0 depuis chaque source 20-80k) vs rivière SWA (merge).
Question : le cooldown réel (annealing) dépasse-t-il le merge post-hoc à chaque point d'entraînement ?
Lit cooldown_profile.csv + les fonds SWA des fenêtres joint (joint_swa_*/r.csv, x = fin de fenêtre).
  venv312/bin/python experiments/can/85_plot_cooldown_profile.py
"""
import csv, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs/can"

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,c-h)*100, min(1.,c+h)*100

# --- cooldown (LR=0) ---
cx, cy, clo, chi = [], [], [], []
prof = f"{R}/cooldown_profile.csv"
if os.path.exists(prof):
    for r in csv.DictReader(open(prof)):
        k=int(float(r["source_k"])); rate=float(r["success_rate"])*100
        lo=float(r.get("ci95_low") or 0)*100; hi=float(r.get("ci95_high") or 0)*100
        n=int(float(r.get("n") or 100))
        if hi<=lo: lo,hi=wilson(round(rate/100*n),n)
        cx.append(k); cy.append(rate); clo.append(rate-lo); chi.append(hi-rate)

# --- merge SWA : merges denses (merge_profile.csv) prioritaires, complétés par les fenêtres rivière ---
WIN = [("joint_swa_W1_12_20k",20),("joint_swa_W2_22_30k",30),("joint_swa_W3_32_40k",40),
       ("joint_swa_W4_42_50k",50),("joint_swa_e56_52_60k",60),("joint_swa_e66_62_70k",70),
       ("joint_swa_e76_72_80k",80)]
def _ci(r, dn=100):  # (rate%, lo%, hi%) d'une ligne r.csv, Wilson en secours
    rate=float(r["success_rate"])*100
    lo=float(r.get("ci95_low") or 0)*100; hi=float(r.get("ci95_high") or 0)*100
    n=int(float(r.get("n") or dn))
    if hi<=lo: lo,hi=wilson(round(rate/100*n), n)
    return rate, lo, hi

mdict = {}
for d, k in WIN:
    p=f"{R}/{d}/r.csv"
    if os.path.exists(p):
        rows=list(csv.DictReader(open(p)))
        if rows: mdict[k]=_ci(rows[-1], 50)
mp=f"{R}/merge_profile.csv"   # merges denses (écrasent les fenêtres rivière au même step)
if os.path.exists(mp):
    for r in csv.DictReader(open(mp)): mdict[int(float(r["source_k"]))]=_ci(r, 100)
sx=sorted(mdict)
sy=[mdict[k][0] for k in sx]; slo=[mdict[k][0]-mdict[k][1] for k in sx]; shi=[mdict[k][2]-mdict[k][0] for k in sx]

# --- brut constant (le rebond, non traité) ---
rawx, rawy, rlo, rhi = [], [], [], []
rr = f"{R}/joint_r34_bigunet/rollouts_50.csv"
if os.path.exists(rr):
    raw = {int(float(r["step"])): r for r in csv.DictReader(open(rr))}
    for k in [20, 30, 40, 50, 60, 70, 80]:
        if k*1000 in raw:
            ra, lo, hi = _ci(raw[k*1000], 50)
            rawx.append(k); rawy.append(ra); rlo.append(ra-lo); rhi.append(hi-ra)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axhline(81.6, color="#27ae60", ls="--", lw=1.4, alpha=0.7, label="SWA late10 (10 ckpts) = 81,6 % @500")
if rawx:
    ax.errorbar(rawx, rawy, yerr=[rlo, rhi], fmt="--^", color="#b0b0b0", ms=7, lw=1.4, alpha=0.85,
                capsize=3, zorder=4, label="brut constant — le REBOND (1 ckpt, n=50, IC95)")
if sx:
    ax.errorbar(sx, sy, yerr=[slo, shi], fmt="-s", color="#c0392b", ms=7, lw=1.8, capsize=4,
                label="fond SWA — fenêtre 5 ckpts (le MERGE, IC95)")
if cx:
    z=sorted(zip(cx,cy,clo,chi)); cx,cy,clo,chi=[list(t) for t in zip(*z)]
    ax.errorbar(cx, cy, yerr=[clo,chi], color="#2c3e50", marker="o", ms=9, lw=2.4, capsize=5,
                zorder=6, label="cooldown LR=0 (annealing réel, n=100)")
    for x,y in zip(cx,cy):
        ax.annotate(f"{y:.0f}", (x,y), color="#2c3e50", fontsize=9, fontweight="bold",
                    xytext=(0,9), textcoords="offset points", ha="center")

ax.set_xlabel("checkpoint de départ (k steps)", fontsize=12)
ax.set_ylabel("succès (rollouts Can) %", fontsize=12)
ax.set_ylim(0, 100); ax.set_xlim(0, 85); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Joint — 3 niveaux à chaque checkpoint : BRUT (rebond) → MERGE (SWA) → COOLDOWN (LR→0)\n"
             "le cooldown (~83%) est stable et au-dessus du merge ET du brut bruité (0-56%) — surtout tôt",
             fontsize=11)
fig.tight_layout(); out=f"{R}/cooldown_profile.png"; fig.savefig(out, dpi=130)
print("Sauvé :", out, "| cooldown :", ", ".join(f"{x}k={y:.0f}" for x,y in zip(cx,cy)) if cx else "(vide)")
