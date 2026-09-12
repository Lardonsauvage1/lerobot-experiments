"""Vérification VISUELLE du banc Can-Occluded avant de lancer la moindre éval.

Contrôle trois choses :
  1. le rendu « sans canette » enlève bien la canette et RIEN d'autre ;
  2. l'état de la simulation est restauré à l'identique après le rendu (aucune fuite) ;
  3. la condition géométrique d'occlusion se déclenche là où on l'attend.

Sortie : results/runs/can/occluded/test_occlusion.png
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

from pathlib import Path
import numpy as np
from PIL import Image

from src import can_eval, can_occlusion

OUT = Path("results/runs/can/occluded"); OUT.mkdir(parents=True, exist_ok=True)
SIZE = 256  # grand pour l'inspection humaine (l'éval tourne à 96)

env = can_eval.make_env()
split = can_eval.load_or_make_split()
states = can_eval.load_init_states(split["val"][:3])

rows = []
for ep in range(3):
    obs = env.reset_to(dict(states=states[ep]))

    qpos_before = np.array(env.env.sim.data.qpos, copy=True)
    qvel_before = np.array(env.env.sim.data.qvel, copy=True)

    normal = can_occlusion.render_normal(env, SIZE, SIZE)
    hidden = can_occlusion.render_without_can(env, SIZE, SIZE)

    dq = np.abs(env.env.sim.data.qpos - qpos_before).max()
    dv = np.abs(env.env.sim.data.qvel - qvel_before).max()
    diff = np.abs(normal.astype(int) - hidden.astype(int)).sum(axis=2)
    changed = (diff > 10).mean()

    eef = np.asarray(obs["robot0_eef_pos"]).flatten()
    can = np.asarray(obs["object"]).flatten()[7:10]
    d_xy = float(np.linalg.norm(eef[:2] - can[:2]))

    print(f"ép {ep}: restauration qpos Δ={dq:.2e} qvel Δ={dv:.2e} | "
          f"pixels changés {changed:.2%} | dist eef-can XY {d_xy:.3f} m "
          f"| eef au-dessus: {eef[2] > can[2]}", flush=True)
    assert dq < 1e-9 and dv < 1e-9, "FUITE : l'état de la sim n'est pas restauré !"
    assert 0.0005 < changed < 0.15, f"pixels changés suspect ({changed:.2%})"

    # bande : normal | sans canette | carte des différences
    dmap = np.stack([np.clip(diff, 0, 255).astype(np.uint8)] * 3, axis=2)
    rows.append(np.concatenate([normal, hidden, dmap], axis=1))

# contrôle de la condition géométrique aux différents rayons
print("\nCondition d'occlusion à l'état initial (bras loin, on attend False partout) :")
for r in [0, 0.03, 0.08, 0.20, 1e9]:
    print(f"  radius={r:<8} -> {can_occlusion.can_is_occluded(obs, r)}")

img = Image.fromarray(np.concatenate(rows, axis=0))
img.save(OUT / "test_occlusion.png")
print(f"\n✓ écrit {OUT/'test_occlusion.png'}  (colonnes : normal | sans canette | différence)")
