"""Phase 4 — vidéos des démonstrations expertes (rejeu des états enregistrés).

Rend en MP4 quelques démos du val set en remettant l'env sur chaque état enregistré
(reset_to(states[t]) + render HD). Mêmes épisodes que 53_record_videos → comparaison
visuelle démo vs modèle sur la même scène.

Sortie : results/runs/lift/51_unet_sweep_eval/videos/DEMO_ep<idx>.mp4
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np

from src import lift_eval

OUT = Path("results/runs/lift/51_unet_sweep_eval/videos")
HD = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    env = lift_eval.make_env()

    with h5py.File(lift_eval.HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for i in val_idx:
            states = f["data"][demos[i]]["states"][:]
            frames = []
            for t in range(len(states)):
                env.reset_to(dict(states=states[t]))
                frames.append(env.env.sim.render(height=HD, width=HD, camera_name="agentview")[::-1])
            out = OUT / f"DEMO_ep{i}.mp4"
            imageio.mimsave(out, frames, fps=20)
            print(f"  démo {i}: {len(frames)} frames -> {out}", flush=True)

    print(f"\nVidéos démos dans : {OUT}")


if __name__ == "__main__":
    main()
