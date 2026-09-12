#!/usr/bin/env python
"""Graphe de l'étape 0 — courbe de robustesse à l'occlusion de la cible.

Quatre panneaux :
  1. succès vs rayon d'occlusion, avec IC95 de Wilson (+ repère du témoin)
  2. la BOUCLE : cycles descente/remontée par épisode, réussis vs ratés
     ⭐ recalculés depuis les trajectoires enregistrées, car `n_approaches` (la métrique
     d'origine) est aveugle : le modèle qui boucle ne SORT jamais de la zone de saisie.
     cf. 115_diag_failure_mode.py et l'angle mort documenté dans src/can_occlusion.py.
  3. le modèle VISE-T-IL JUSTE ? distance XY minimale atteinte, réussis vs ratés
     — c'est la mesure qui dit si l'échec est perceptif (il sait où elle est mais rate la
     préhension) ou décisionnel (il ne sait pas où aller).
  4. temps passé sans voir la cible, et fraction d'épisodes réellement occlus

Sortie : results/runs/can/occluded/occlusion_curve.png
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.can_occlusion import z_cycles

OCC = Path("results/runs/can/occluded")
CURVE = OCC / "occlusion_curve.json"


def traj_stats(radius):
    """Recalcule depuis le .npz ce que la sonde d'origine ne mesurait pas."""
    tag = "inf" if radius >= 1e3 else f"{int(round(radius * 100)):02d}"
    f = OCC / f"traj_r{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    pro, can = d["proprio"], d["can_pos"]
    bounds = list(d["ep_start"]) + [len(pro)]
    meta = json.loads(str(d["ep_meta"]))
    out = {"ok_cyc": [], "ko_cyc": [], "ok_dxy": [], "ko_dxy": []}
    for e, m in enumerate(meta):
        lo, hi = bounds[e], bounds[e + 1]
        eef, cp = pro[lo:hi, :3], can[lo:hi]
        c = z_cycles(eef[:, 2])
        dxy = float(np.linalg.norm(eef[:, :2] - cp[:, :2], axis=1).min())
        (out["ok_cyc"] if m["success"] else out["ko_cyc"]).append(c)
        (out["ok_dxy"] if m["success"] else out["ko_dxy"]).append(dxy)
    return out


res = sorted(json.loads(CURVE.read_text())["results"], key=lambda r: r["radius"])
xs = [r["radius"] * 100 if r["radius"] < 1e3 else None for r in res]
INF_X = max([x for x in xs if x is not None], default=4) * 1.6   # ∞ placé hors échelle
xs = [x if x is not None else INF_X for x in xs]

fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

# --- 1. succès
sr = [r["success_rate"] * 100 for r in res]
lo = [(r["success_rate"] - r["ci95"][0]) * 100 for r in res]
hi = [(r["ci95"][1] - r["success_rate"]) * 100 for r in res]
ax[0, 0].errorbar(xs, sr, yerr=[lo, hi], marker="o", capsize=4, lw=2)
ax[0, 0].axhline(sr[0], color="grey", ls="--", lw=1, label=f"témoin {sr[0]:.1f} %")
ax[0, 0].set_xlabel("rayon d'occlusion (cm)   —   dernier point = jamais visible")
ax[0, 0].set_ylabel("succès (%)"); ax[0, 0].set_ylim(0, 100)
ax[0, 0].set_title("Chute du succès quand la cible disparaît"); ax[0, 0].legend()
for x, y, r in zip(xs, sr, res):
    ax[0, 0].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=8)

# --- 2. la boucle
st = {r["radius"]: traj_stats(r["radius"]) for r in res}
have = [(x, r) for x, r in zip(xs, res) if st[r["radius"]]]
if have:
    w = max(min(np.diff(sorted(set(x for x, _ in have))), default=0.6) * 0.35, 0.12)
    ax[0, 1].bar([x - w / 2 for x, _ in have],
                 [np.mean(st[r["radius"]]["ok_cyc"] or [0]) for _, r in have],
                 width=w, label="réussis", color="tab:green")
    ax[0, 1].bar([x + w / 2 for x, _ in have],
                 [np.mean(st[r["radius"]]["ko_cyc"] or [0]) for _, r in have],
                 width=w, label="ratés", color="tab:red")
    ax[0, 1].set_ylabel("cycles descente/remontée par épisode")
    ax[0, 1].set_xlabel("rayon d'occlusion (cm)")
    ax[0, 1].set_title("La BOUCLE du robot réel\n(rater → remonter → redescendre)")
    ax[0, 1].legend()

    # --- 3. vise-t-il juste ?
    ax[1, 0].bar([x - w / 2 for x, _ in have],
                 [np.mean(st[r["radius"]]["ok_dxy"] or [0]) * 100 for _, r in have],
                 width=w, label="réussis", color="tab:green")
    ax[1, 0].bar([x + w / 2 for x, _ in have],
                 [np.mean(st[r["radius"]]["ko_dxy"] or [0]) * 100 for _, r in have],
                 width=w, label="ratés", color="tab:red")
    ax[1, 0].set_ylabel("distance XY minimale atteinte (cm)")
    ax[1, 0].set_xlabel("rayon d'occlusion (cm)")
    ax[1, 0].set_title("Le modèle vise-t-il juste ?\n(proche = il sait où elle est, il rate la préhension)")
    ax[1, 0].legend()

# --- 4. exposition à l'occlusion
ax[1, 1].plot(xs, [r["occl_fraction_mean"] * 100 for r in res], marker="o",
              label="temps sans voir la cible")
ax[1, 1].plot(xs, [r["n_episodes_occluded"] / r["n"] * 100 for r in res], marker="s",
              label="épisodes touchés")
ax[1, 1].set_ylabel("%"); ax[1, 1].set_xlabel("rayon d'occlusion (cm)")
ax[1, 1].set_title("Exposition à l'occlusion"); ax[1, 1].legend(); ax[1, 1].set_ylim(0, 105)

n = res[0]["n"]
fig.suptitle(f"Étape 0 — banc Can-Occluded · modèle témoin markovien (n_obs_steps=2, "
             f"même structure que le modèle réel) · {n} rollouts/point", fontsize=12)
fig.tight_layout()
fig.savefig(OCC / "occlusion_curve.png", dpi=110)
print(f"✓ {OCC / 'occlusion_curve.png'}")
for x, r in zip(xs, res):
    s = st[r["radius"]]
    cyc = f"{np.mean(s['ko_cyc']):.2f}" if s and s["ko_cyc"] else "-"
    print(f"  r={r['radius']:<7} {r['success_rate']:6.1%} [{r['ci95'][0]:.1%}-{r['ci95'][1]:.1%}]"
          f" | cycles/échec {cyc}")
