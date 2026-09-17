#!/usr/bin/env python
"""Génère un dataset Can = 150 démos expertes + N épisodes de RATTRAPAGE scriptés.

Pourquoi. Les 200 démonstrations de Robomimic sont toutes des réussites du premier coup :
la pince ne s'y referme JAMAIS dans le vide. La situation « je viens de rater » n'existe donc
pas dans le monde du modèle, et face à elle en boucle fermée il invente — c'est la « saisie
fantôme » filmée dans cam2_E_birdview_wrist/videos/E_ep000_ECHEC.mp4, où il se place
parfaitement, referme, et repart les mains vides sans jamais le constater.

Ce que contient un épisode de rattrapage : approche, décalage de 4,5 cm, descente, fermeture
DANS LE VIDE, réouverture, remontée, recentrage, descente, saisie, transport, dépôt.

Deux garde-fous (voir src/can_scripted.py) :
  - si la fermeture « à vide » a réellement attrapé, l'épisode est JETÉ — lui faire rouvrir
    apprendrait à lâcher une prise réussie ;
  - filtre indépendant : la canette ne doit se soulever qu'UNE fois (comptage des fronts).

Seules les réussites sont conservées.

Usage : venv312/bin/python -u experiments/can/139_gen_rattrapage.py --n 75
"""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np

from src import can_eval, can_scripted as SC

spec = importlib.util.spec_from_file_location("v", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

IMG = 96
OUT = Path("data_cache/lerobot_can_ph_proprio_correction")
REPO = "local/can_ph_proprio_correction"
N_EXPERT = 150


def state9(obs):
    o = lambda k: np.asarray(obs[k]).flatten()
    return np.concatenate([o("robot0_eef_pos"), o("robot0_eef_quat"),
                           o("robot0_gripper_qpos")]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75, help="épisodes de rattrapage à CONSERVER")
    ap.add_argument("--offset", type=float, default=0.045)
    a = ap.parse_args()

    if OUT.exists():
        print(f"⚠️  {OUT} existe déjà — supprime-le pour relancer."); return 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    img_feat = {"dtype": "video", "shape": (3, IMG, IMG), "names": ["channels", "height", "width"]}
    names = ["eef_pos_x","eef_pos_y","eef_pos_z","eef_quat_w","eef_quat_x","eef_quat_y",
             "eef_quat_z","gripper_l","gripper_r"]
    ds = LeRobotDataset.create(repo_id=REPO, fps=20, root=OUT, features={
        "observation.image": img_feat,
        "observation.state": {"dtype": "float32", "shape": (9,), "names": names},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["dx","dy","dz","drx","dry","drz","gripper"]}})

    env = can_eval.make_env()
    sim = env.env.sim
    shot = lambda: sim.render(height=IMG, width=IMG, camera_name="agentview")[::-1]
    TASK = "pick the can and place it in the bin"

    # --- 1) les 150 démonstrations expertes, rejouées à l'identique ---
    print(f"[1/2] {N_EXPERT} démonstrations expertes...", flush=True)
    with h5py.File(can_eval.HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k, d in enumerate(demos[:N_EXPERT]):
            st = f["data"][d]["states"][:]; ac = f["data"][d]["actions"][:]
            o = f["data"][d]["obs"]
            s9 = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                                 o["robot0_gripper_qpos"][:]], axis=1).astype(np.float32)
            env.reset()
            for i in range(st.shape[0]):
                sim.set_state_from_flattened(st[i]); sim.forward()
                ds.add_frame({"observation.image": np.ascontiguousarray(shot().transpose(2,0,1)),
                              "observation.state": s9[i], "action": ac[i].astype(np.float32),
                              "task": TASK})
            ds.save_episode()
            if (k+1) % 30 == 0: print(f"      {k+1}/{N_EXPERT}", flush=True)

    # --- 2) les rattrapages ---
    print(f"[2/2] rattrapages (objectif {a.n} conservés)...", flush=True)
    states = V.make_eval_states()
    rng = np.random.default_rng(12345)
    kept = tried = thrown = failed = 0
    while kept < a.n and tried < a.n * 4:
        i = tried % len(states)
        off = rng.uniform(-1, 1, 3) * np.array([a.offset, a.offset, 0.0])
        frames = []
        orig = env.step
        def rec(act):
            frames.append(np.ascontiguousarray(shot().transpose(2, 0, 1)))
            return orig(act)
        env.step = rec
        try:
            r, acts, obss, cut = SC.scripted_episode(env, states[i], perturb=off, seed=tried)
        finally:
            env.step = orig
        tried += 1
        if r is None: thrown += 1; continue
        if not r:     failed += 1; continue
        # ⭐ On NE GARDE QUE LA CORRECTION : tout ce qui précède est le trajet volontaire
        # vers l'erreur, et l'apprendre reviendrait à enseigner au modèle à aller se tromper.
        # C'est ce qui a fait échouer la première version (R1 : 66,8 % contre 78,5 %).
        n = min(len(frames), len(acts), len(obss))
        for j in range(cut, n):
            ds.add_frame({"observation.image": frames[j],
                          "observation.state": state9(obss[j]),
                          "action": acts[j], "task": TASK})
        ds.save_episode(); kept += 1
        if kept % 10 == 0:
            print(f"      {kept}/{a.n} gardés  ({tried} essais, {thrown} jetés, {failed} ratés)", flush=True)

    print(f"\n✓ {OUT}")
    print(f"  {N_EXPERT} expertes + {kept} rattrapages = {ds.num_episodes} épisodes, {ds.num_frames} frames")
    print(f"  taux de conservation : {kept}/{tried}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
