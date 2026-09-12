"""Précalcule les stats des actions RELATIVES chunk-wise + valide l'amplitude vs séquentiel.

Chunk-wise : rel[j] = action[i + (j - (n_obs-1))][:7] - state[i][:7]  (ancre = state au step courant i).
Aligné sur la fenêtre diffusion LeRobot (n_obs=2, horizon=16 -> offsets -1..14).
Sort : stats min/max/mean/std relatives (7 joints) -> JSON, + compare l'amplitude au séquentiel.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = "data_cache/lerobot_can_ph_joint_birdview"
N_OBS, HORIZON = 2, 16
OUT = "data_cache/lerobot_can_ph_joint_birdview/meta/relstats_chunkwise.json"

# charge action + state par épisode via les parquet LeRobot
import pandas as pd, glob
files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
print(f"{len(files)} fichiers parquet")

rel_all = []          # tous les deltas relatifs chunk-wise (7 joints)
seq_all = []          # deltas séquentiels (pour comparaison)
offsets = list(range(-(N_OBS - 1), HORIZON - (N_OBS - 1)))  # -1 .. 14
print("offsets horizon (relatifs à l'ancre) :", offsets[0], "..", offsets[-1], f"({len(offsets)} = horizon)")

for fp in files:
    df = pd.read_parquet(fp, columns=["action", "observation.state", "episode_index"])
    for ep, g in df.groupby("episode_index"):
        act = np.stack(g["action"].values)[:, :7]          # (T,7) cibles articulaires absolues
        st = np.stack(g["observation.state"].values)[:, :7]  # (T,7) jpos
        T = len(act)
        seq_all.append(np.diff(st, axis=0))                 # séquentiel = jpos[t+1]-jpos[t]
        for i in range(T):
            anchor = st[i]
            for off in offsets:
                j = i + off
                if 0 <= j < T:
                    rel_all.append(act[j] - anchor)         # cible[j] - ancre[i]

rel = np.array(rel_all); seq = np.concatenate(seq_all, axis=0)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print(f"\n=== stats RELATIVES chunk-wise (rad, 7 joints) sur {len(rel)} échantillons ===")
stats = {"min": rel.min(0).tolist(), "max": rel.max(0).tolist(),
         "mean": rel.mean(0).tolist(), "std": rel.std(0).tolist()}
for i in range(7):
    print(f"  j{i}: std={rel[:,i].std():.4f}  min/max=[{rel[:,i].min():.3f},{rel[:,i].max():.3f}]")

print(f"\n=== AMPLITUDE : chunk-wise vs séquentiel (le test clé) ===")
print(f"  {'joint':<6}{'chunk std':>11}{'seq std':>10}{'ratio':>8}")
for i in range(7):
    cs, ss = rel[:, i].std(), seq[:, i].std()
    print(f"  j{i:<5}{cs:>11.4f}{ss:>10.4f}{cs/ss:>8.1f}x")
print(f"\n  -> chunk-wise moyen = {rel.std(0).mean():.4f} rad vs séquentiel {seq.std(0).mean():.4f} rad "
      f"(x{rel.std(0).mean()/seq.std(0).mean():.0f})")
print("  -> si x>>1 : deltas chunk-wise BIEN plus grands -> bien moins sujets à l'effondrement de magnitude")

Path(OUT).write_text(json.dumps(stats, indent=2))
print(f"\nStats relatives sauvées : {OUT}")
