#!/usr/bin/env python
"""Diagnostic : le banc Can-Occluded reproduit-il VRAIMENT le mode d'échec du robot réel ?

Sur le vrai bras (note atomman §2), l'échec est une BOUCLE :
    rater -> reculer -> ne plus la voir -> redescendre -> rater -> ...
En simulation à 4 cm, la sonde compte 1,09 approche par échec contre 1,03 au témoin :
autant dire aucune boucle. Deux explications possibles, et il faut trancher —

  (A) la sonde est AVEUGLE : ses seuils (XY < 6 cm, dz < 5 cm) sont trop stricts, et le
      modèle boucle plus loin ou plus haut sans être compté ;
  (B) le modèle ne boucle réellement pas : il échoue autrement (se fige, dérive, n'approche
      jamais), et le banc mesure alors une fragilité à l'occlusion qui n'est PAS celle du
      robot réel.

On tranche en comptant les cycles descente/remontée de l'effecteur INDÉPENDAMMENT de toute
proximité à la canette — un critère que (A) ne peut pas biaiser.

Lancer : venv312/bin/python -u experiments/can/115_diag_failure_mode.py
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np

OCC = Path("results/runs/can/occluded")


def z_cycles(z, min_amp=0.02):
    """Nombre de cycles descente->remontée d'amplitude > min_amp (m). Sans référence à la
    canette : capture « il remonte et redescend » même si le bras vise complètement à côté."""
    ext, direction = [z[0]], 0
    for v in z[1:]:
        if direction >= 0 and v < ext[-1] - min_amp:
            ext.append(v); direction = -1
        elif direction <= 0 and v > ext[-1] + min_amp:
            ext.append(v); direction = 1
        elif (direction < 0 and v < ext[-1]) or (direction > 0 and v > ext[-1]):
            ext[-1] = v
    return max((len(ext) - 1) // 2, 0)


for f in sorted(OCC.glob("traj_r*.npz")):
    d = np.load(f, allow_pickle=True)
    pro, can, act = d["proprio"], d["can_pos"], d["action"]
    starts, meta = d["ep_start"], json.loads(str(d["ep_meta"]))
    bounds = list(starts) + [len(pro)]
    rows = []
    for e, m in enumerate(meta):
        lo, hi = bounds[e], bounds[e + 1]
        eef, cp = pro[lo:hi, :3], can[lo:hi]
        dxy = np.linalg.norm(eef[:, :2] - cp[:, :2], axis=1)
        rows.append({
            "success": m["success"],
            "T": hi - lo,
            "dxy_min": float(dxy.min()),
            "dxy_end": float(dxy[-1]),
            "z_cycles": z_cycles(eef[:, 2]),
            # combien de fois le bras descend près de la canette, à seuils ÉLARGIS
            "near_10cm": int(((dxy < 0.10) & (eef[:, 2] - cp[:, 2] < 0.10)).sum()),
            "grip_close": int((np.diff((act[lo:hi, 6] > 0).astype(int)) == 1).sum()),
            "path": float(np.abs(np.diff(eef, axis=0)).sum()),
        })
    ok = [r for r in rows if r["success"]]
    ko = [r for r in rows if not r["success"]]
    print(f"\n===== {f.name}  ({len(ok)} réussis / {len(ko)} ratés) =====")
    print(f"{'':>18}{'RÉUSSIS':>12}{'RATÉS':>12}")
    for k, lab, fmt in [("z_cycles", "cycles haut/bas", "{:.2f}"),
                        ("grip_close", "fermetures pince", "{:.2f}"),
                        ("dxy_min", "dist. XY min (m)", "{:.3f}"),
                        ("dxy_end", "dist. XY finale (m)", "{:.3f}"),
                        ("near_10cm", "pas près (10cm)", "{:.0f}"),
                        ("path", "chemin parcouru (m)", "{:.2f}"),
                        ("T", "durée (pas)", "{:.0f}")]:
        a = np.mean([r[k] for r in ok]) if ok else float("nan")
        b = np.mean([r[k] for r in ko]) if ko else float("nan")
        print(f"{lab:>18}{fmt.format(a):>12}{fmt.format(b):>12}")
    if ko:
        c = np.array([r["z_cycles"] for r in ko])
        print(f"   ratés avec >=2 cycles haut/bas : {(c >= 2).mean():.0%}  "
              f"| >=3 : {(c >= 3).mean():.0%}  (= la boucle du robot réel)")
        n = np.array([r["near_10cm"] for r in ko])
        print(f"   ratés n'ayant JAMAIS approché à 10 cm : {(n == 0).mean():.0%}")
