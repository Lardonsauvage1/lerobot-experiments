"""49 (18/11, Sam) — sous-jeu d'un jeu café 10 Hz : ne garde que les épisodes dont la stratégie du tiroir est dans --garder.
Contrôle de l'effet « poison » : FP50V (40 S3 + 40 S1/S2/S4) -> FS50V = les 40 démos S1/S2/S4 SEULES (mêmes démos, mêmes départs).
Copie aussi la banque de biais (par image) en ne gardant que les images des épisodes gardés, dans le même ordre.
  python experiments/cafe/49_sous_jeu_strat.py --source cafeFP50V80_fermer_tiroir --sortie cafeFS50V80_fermer_tiroir --garder 1,2,4
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True); ap.add_argument("--sortie", required=True); ap.add_argument("--garder", required=True)
a = ap.parse_args()
dc = Path("data_cache")
garder = {int(x) for x in a.garder.split(",")}
ch = json.load(open(dc / a.source / "cafe_demos.json"))["chaines"]
strat = [c["gestes"][list(c["gestes"])[0]]["strat"] for c in ch]
src, dst = dc / f"{a.source}_10hz", dc / f"{a.sortie}_10hz"
assert not dst.exists(), f"{dst} existe déjà"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
ds = LeRobotDataset(f"local/{a.source}_10hz", root=src)
assert ds.meta.total_episodes == len(strat), (ds.meta.total_episodes, len(strat))
# images par épisode, dans l'ordre global du jeu (pour la banque)
ep_idx = np.asarray(ds.hf_dataset["episode_index"])
assert (np.diff(ep_idx) >= 0).all(), "images non triées par épisode"
retirer = [e for e, s in enumerate(strat) if s not in garder]
gardes = [e for e, s in enumerate(strat) if s in garder]
print(f"stratégies {strat}\n  garder {sorted(garder)} : {len(gardes)} épisodes, retirer {len(retirer)}", flush=True)
nouveau = delete_episodes(ds, episode_indices=retirer, output_dir=dst, repo_id=f"local/{a.sortie}_10hz")
bq = np.load(dc / f"banque_biais_{a.source}_10hz.npy")
assert bq.shape[0] == len(ep_idx), (bq.shape, len(ep_idx))
masque = np.isin(ep_idx, gardes)
np.save(dc / f"banque_biais_{a.sortie}_10hz.npy", bq[masque])
assert nouveau.meta.total_frames == int(masque.sum()), (nouveau.meta.total_frames, int(masque.sum()))
# trace des démos gardées (pour les contrôles et la lecture)
(dst / "sous_jeu.json").write_text(json.dumps({"source": a.source, "garder": sorted(garder), "episodes_source": gardes,
                                                "strat": [strat[e] for e in gardes]}, indent=1))
print(f"SOUS-JEU DONE {dst} : {nouveau.meta.total_episodes} épisodes, {nouveau.meta.total_frames} images ; banque {int(masque.sum())}", flush=True)
