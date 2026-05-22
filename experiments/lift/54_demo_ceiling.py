"""Phase 4 — plafond théorique : performance des démos expertes sur les 50 scènes val.

Pour chaque démo val, on remet l'env sur CHAQUE état enregistré (reset_to(states[t])) et on
mesure avec le MÊME critère que nos modèles (env.is_success(), cube z) → la perf de l'expert
humain, qui est le maximum théorique qu'une politique d'imitation peut viser. Pas de replay
d'actions (donc pas de divergence de dynamique) : on lit la trajectoire enregistrée.

Comparable directement aux résultats de 52_sweep_eval (mêmes 50 épisodes, mêmes métriques).
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
from pathlib import Path

import h5py
import numpy as np

from src import lift_eval

MAX_STEPS = 200
OUT = Path("results/runs/lift/51_unet_sweep_eval/demo_ceiling.json")


def main():
    split = lift_eval.load_or_make_split()
    val_idx = split["val"]
    env = lift_eval.make_env()

    per_ep = []
    with h5py.File(lift_eval.HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for i in val_idx:
            states = f["data"][demos[i]]["states"][:]
            T = min(len(states), MAX_STEPS)
            max_z, t_success, flags = 0.0, None, []
            for t in range(T):
                obs = lift_eval.fix_obs_sign(env.reset_to(dict(states=states[t])))
                max_z = max(max_z, float(np.asarray(obs["object"])[2]))
                s = bool(env.is_success()["task"])
                flags.append(s)
                if s and t_success is None:
                    t_success = t
            success = t_success is not None
            hold = float(np.mean(flags[t_success:])) if success else 0.0
            per_ep.append({"ep": int(i), "len": int(T), "success": success,
                           "t_success": t_success if success else None,
                           "max_z": max_z, "hold_fraction": hold})
            print(f"  démo {i}: {'✓' if success else '✗'} t_succ={t_success} max_z={max_z:.3f} hold={hold:.2f} (len {T})", flush=True)

    agg = lift_eval.aggregate(per_ep)
    print("\n=== PLAFOND DÉMOS (expert, 50 scènes val) ===")
    print(f"  success_rate      : {agg['success_rate']:.0%}")
    print(f"  t_success médian  : {agg['t_success_median']}")
    print(f"  max_z moyen       : {agg['max_z_mean']:.3f}")
    print(f"  hold_fraction moy : {agg['hold_fraction_mean']:.2f}")
    OUT.write_text(json.dumps({"n": len(val_idx), "aggregate": agg, "per_episode": per_ep}, indent=2))
    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
