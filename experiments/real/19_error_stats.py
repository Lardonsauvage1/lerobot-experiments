#!/usr/bin/env python
"""Precision du modele sur des episodes complets (teacher-forcing, 1 prediction fraiche/frame).
Pour chaque frame : obs (images+pose reelles du dataset) -> prediction ; comparee a la vraie action.
Sort ecart-type et ecart max de l'erreur, en DEGRES (par joint + global) + fiabilite gripper."""
import sys, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/runs/real/apple_joint_224_r34/cooldown_012000/checkpoints/005000/pretrained_model"
EPISODES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5, 8, 22]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

p = DiffusionPolicy.from_pretrained(MODEL).eval().to(DEV)
p.config.n_action_steps = 1          # 1 -> re-echantillonne a CHAQUE frame (prediction fraiche)
pre, post = make_pre_post_processors(policy_cfg=p.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": DEV.type}})
IMG = [k for k in p.config.input_features if "image" in k]

all_err = []      # erreurs joints (rad, signees) [N,5]
grip_ok = 0; grip_tot = 0
for ep in EPISODES:
    ds = LeRobotDataset("local/apple_joint_224", root="data_cache/lerobot_apple_joint_224", episodes=[ep])
    p.reset()
    n = ds.num_frames
    for t in range(n):
        f = ds[t]
        obs = {k: f[k].unsqueeze(0).to(DEV) for k in IMG}
        obs["observation.state"] = f["observation.state"].unsqueeze(0).to(DEV)
        with torch.no_grad():
            a = post(p.select_action(pre({k: v.clone() for k, v in obs.items()}))).squeeze(0).cpu().numpy()
        gt = f["action"].numpy()
        all_err.append(a[:5] - gt[:5])
        grip_ok += int((a[5] > 0.5) == (gt[5] > 0.5)); grip_tot += 1
    print(f"ep{ep}: {n} frames traitées")

E = np.rad2deg(np.array(all_err))        # [N,5] en degres
absE = np.abs(E)
print(f"\n===== {len(E)} frames sur épisodes {EPISODES} — erreur prédiction vs vérité (degrés) =====")
print(f"{'joint':>8} | {'moy|err|':>9} | {'ecart-type':>10} | {'max|err|':>9}")
for j in range(5):
    print(f"joint_{j+1:>2} | {absE[:,j].mean():>9.2f} | {E[:,j].std():>10.2f} | {absE[:,j].max():>9.2f}")
print(f"{'GLOBAL':>8} | {absE.mean():>9.2f} | {E.std():>10.2f} | {absE.max():>9.2f}")
print(f"\n>>> ÉCART-TYPE global = {E.std():.2f}°   |   ÉCART MAX = {absE.max():.2f}°")
print(f">>> Gripper correct : {100*grip_ok/grip_tot:.1f}% ({grip_ok}/{grip_tot})")
