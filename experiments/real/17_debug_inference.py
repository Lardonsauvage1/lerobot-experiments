#!/usr/bin/env python
"""Diagnostic : le modele B (cooldown@12000) genere-t-il des coordonnees COHERENTES ?
Teacher-forcing : on nourrit les VRAIES images/joints du dataset (ep 22) frame par frame,
et on compare l'action predite a la verite terrain de la demo (action du dataset).
- Si erreur petite -> modele coherent offline (probleme robot = pretraitement au deploiement).
- Si erreur enorme -> modele/entree incoherents.
"""
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

import sys
# argv[1] = chemin du modele (defaut Mac ; sur atomman : deployable_models/apple/B_cd5k_from12000/brut)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/runs/real/apple_joint_224_r34/cooldown_012000/checkpoints/005000/pretrained_model"
# argv[2] = root du dataset (defaut Mac ; sur atomman : data_cache/lerobot_apple_joint_224)
DS_ROOT = sys.argv[2] if len(sys.argv) > 2 else "data_cache/lerobot_apple_joint_224"
EP, NSTEPS = 22, 16
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ds = LeRobotDataset("local/apple_joint_224", root=DS_ROOT, episodes=[EP])
policy = DiffusionPolicy.from_pretrained(MODEL); policy.eval().to(DEV); policy.reset()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": DEV.type}})
IMG = [k for k in policy.config.input_features if "image" in k]

print(f"modele B (cd@12000) | ep {EP} | teacher-forcing sur {NSTEPS} frames | device {DEV.type}")
print(f"\n{'t':>2} | {'STATE courant (5 joints)':^42} | grip")
print(f"   | {'ACTION predite (5 joints)':^42} | grip")
print(f"   | {'VERITE dataset (5 joints)':^42} | grip   | max|err joints|")
print("-"*104)
errs = []
for t in range(NSTEPS):
    f = ds[t]
    obs = {k: f[k].unsqueeze(0).to(DEV) for k in IMG}
    obs["observation.state"] = f["observation.state"].unsqueeze(0).to(DEV)
    with torch.no_grad():
        # ⚠️ INDISPENSABLE : pre() normalise l'obs. Sans lui -> sortie constante ~1 rad a cote (bug robot).
        a = post(policy.select_action(pre({k: v.clone() for k, v in obs.items()}))).squeeze(0).cpu().numpy()
    gt = f["action"].numpy(); st = f["observation.state"].numpy()
    e = np.abs(a[:5] - gt[:5]).max(); errs.append(e)
    print(f"{t:>2} | {str(np.round(st,3)):^42} | {'':4}")
    print(f"   | {str(np.round(a[:5],3)):^42} | {a[5]:.2f}")
    print(f"   | {str(np.round(gt[:5],3)):^42} | {gt[5]:.2f}   | {e:.3f} rad")
    print()
print(f"=> erreur joints predite-vs-verite : moyenne {np.mean(errs):.3f} rad, max {np.max(errs):.3f} rad "
      f"({np.degrees(np.mean(errs)):.1f}° / {np.degrees(np.max(errs)):.1f}°)")
print("Repere : <~0,05 rad (3°) = tres coherent ; >~0,3 rad (17°) = incoherent (bug ou modele KO).")
