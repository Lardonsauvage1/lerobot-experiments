"""Vidéos de démos expertes Can avec les 2 caméras (agentview + wrist) côte à côte.

Permet de voir ce que les modèles 2 cams (06/08) voyaient à l'entraînement.
Replay des états sauvés (HDF5 dataset) dans l'env, double rendu.

Sortie : data_cache/can_demos/can_demo<idx>_2cams.mp4
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

from pathlib import Path

import h5py
import imageio
import numpy as np

HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUT = Path("data_cache/can_demos")
DEMO_IDS = [0, 50, 100]   # 3 démos variées
HD = 256


def main():
    import robosuite as rs
    OUT.mkdir(parents=True, exist_ok=True)
    env = rs.make(env_name="PickPlaceCan", robots="Panda",
                  has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=False, control_freq=20)
    f = h5py.File(HDF5, "r")
    for di in DEMO_IDS:
        d = f["data"][f"demo_{di}"]
        states = d["states"][:]
        env.reset()
        frames = []
        for s in states:
            env.sim.set_state_from_flattened(s); env.sim.forward()
            ag = env.sim.render(height=HD, width=HD, camera_name="agentview")[::-1]
            wr = env.sim.render(height=HD, width=HD, camera_name="robot0_eye_in_hand")[::-1]
            frames.append(np.concatenate([ag, wr], axis=1))  # côte à côte 512×256
        out = OUT / f"can_demo{di}_2cams.mp4"
        imageio.mimsave(out, frames, fps=20)
        print(f"  démo {di}: {len(frames)} frames -> {out}", flush=True)
    env.close()
    print(f"\nVidéos : {OUT}")


if __name__ == "__main__":
    main()
