#!/usr/bin/env python3
"""Graphe 'river valley' du joint : rebond brut + fond SWA par fenêtre.
Chaque fenêtre = barre horizontale (son ÉTENDUE) à la hauteur du fond, annotée (fond% + nb ckpts).
+ axe LR (constant 1e-4) pour matérialiser 'LR plat mais le fond monte'.
Lit les fonds depuis les r.csv des modèles SWA (auto-remplis au fil des évals).
  venv312/bin/python experiments/can/77_plot_river.py
"""
import csv, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, c-h)*100, min(1.0, c+h)*100

RDJ = "results/runs/can/joint_r34_bigunet"

# rebond brut (succès par checkpoint)
raw = {}
for r in csv.DictReader(open(f"{RDJ}/rollouts_50.csv")):
    s = int(float(r["step"]))
    if 2000 <= s <= 100000:
        raw[s] = float(r["success_rate"]) * 100
xs = sorted(raw); ys = [raw[s] for s in xs]

# registre des fenêtres : (label, début_k, fin_k, nb_ckpts, dossier, fond_connu)
WINS = [
    ("W1", 12, 20, 5, "joint_swa_W1_12_20k", 54), ("v21", 16, 24, 5, "joint_swa_v21_16_24k", None),
    ("W2", 22, 30, 5, "joint_swa_W2_22_30k", 70), ("v31", 26, 34, 5, "joint_swa_v31_26_34k", None),
    ("W3", 32, 40, 5, "joint_swa_W3_32_40k", 70), ("v41", 36, 44, 5, "joint_swa_v41_36_44k", None),
    ("W4", 42, 50, 5, "joint_swa_W4_42_50k", 76),
    ("e56", 52, 60, 5, "joint_swa_e56_52_60k", None), ("e66", 62, 70, 5, "joint_swa_e66_62_70k", None),
    ("e76", 72, 80, 5, "joint_swa_e76_72_80k", None),
    ("e86", 82, 90, 5, "joint_swa_e86_82_90k", None), ("e96", 92, 100, 5, "joint_swa_e96_92_100k", None),
    ("late10", 32, 50, 10, "joint_swa_late10", 88), ("wide", 42, 80, 20, "joint_swa_wide_42_80k", None),
]
def floor(d, fallback):
    """retourne (taux%, ic95_lo%, ic95_hi%) ou None."""
    p = f"results/runs/can/{d}/r.csv"
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        if rows:
            r = rows[-1]
            rate = float(r["success_rate"]) * 100
            lo = float(r.get("ci95_low") or 0) * 100
            hi = float(r.get("ci95_high") or 0) * 100
            if hi <= lo:  # CI absent → Wilson (n=50)
                lo, hi = wilson(round(rate/100*50), 50)
            return rate, lo, hi
    if fallback is None:
        return None
    lo, hi = wilson(round(fallback/100*50), 50)
    return float(fallback), lo, hi

fig, ax = plt.subplots(figsize=(13.5, 7))
ax.plot([x/1000 for x in xs], ys, "-o", color="#9db8d8", ms=3, lw=1, label="succès brut par checkpoint (le REBOND, n=50)")
ax.fill_between([x/1000 for x in xs], 0, ys, color="#9db8d8", alpha=0.12)

for label, a, b, n, d, fb in WINS:
    res = floor(d, fb)
    if res is None:
        continue
    f, lo, hi = res
    wide = n >= 10
    col = "#27ae60" if wide else "#c0392b"
    mid = (a + b) / 2
    ax.hlines(f, a, b, color=col, lw=6 if wide else 3.5, alpha=0.9, zorder=5)
    ax.plot([a, b], [f, f], "|", color=col, ms=9, mew=2, zorder=6)  # bornes de la fenêtre
    ax.errorbar(mid, f, yerr=[[f-lo], [hi-f]], color=col, capsize=4, elinewidth=1.4, alpha=0.65, zorder=6)  # IC95
    ax.annotate(f"{f:.0f}% ({n}ck)", (mid, hi), color=col, fontsize=8, fontweight="bold",
                xytext=(0, 4), textcoords="offset points", ha="center", zorder=7)

# axe LR (constant) à droite
ax2 = ax.twinx()
ax2.plot([2, 80], [1e-4, 1e-4], "--", color="#999", lw=1.6)
ax2.set_yscale("log"); ax2.set_ylim(1e-6, 1e-2)
ax2.set_ylabel("learning rate (log)", color="#999"); ax2.tick_params(axis="y", labelcolor="#999")
ax2.annotate("LR CONSTANT 1e-4 — aucun annealing", (41, 1e-4), color="#777", fontsize=9,
             xytext=(0, 6), textcoords="offset points", ha="center")

# légende fenêtres
ax.hlines([], [], [], color="#c0392b", lw=3.5, label="fond SWA — fenêtre 5 ckpts (barre = étendue)")
ax.hlines([], [], [], color="#27ae60", lw=6, label="fond SWA — fenêtre LARGE (≥10 ckpts)")
ax.set_xlabel("step d'entraînement (k)", fontsize=12); ax.set_ylabel("succès (rollouts) %", fontsize=12)
ax.set_ylim(-3, 100); ax.set_xlim(0, 83); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
ax.set_title("Joint (LR CONSTANT) — profil de la « rivière »\nbarre = fenêtre SWA (largeur = étendue, n = nb checkpoints moyennés) ; LR plat mais le fond MONTE puis plafonne", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{RDJ}/river_profile.png", dpi=120)
print(f"Sauvé : {RDJ}/river_profile.png ; fenêtres : " +
      ", ".join(f"{l}={floor(d,fb)[0]:.0f}[{floor(d,fb)[1]:.0f}-{floor(d,fb)[2]:.0f}]"
                for l, a, b, n, d, fb in WINS if floor(d, fb) is not None))
