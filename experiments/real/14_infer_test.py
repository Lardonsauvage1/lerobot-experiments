#!/usr/bin/env python
"""Test d'inference AUTONOME d'un modele "pomme" (a lancer sur atomman, CPU).
Charge un checkpoint deployable + son preprocessor/postprocessor, fabrique une observation
factice (2 cams 224x224 + 5 joints), et produit UNE action 6D (5 joints + gripper).
Prouve que le modele se charge et infere avec tout deja installe.

Usage:
  venv/bin/python experiments/real/14_infer_test.py deployable_models/apple/A_cd5k_from2000/brut
"""
import sys
from pathlib import Path
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "deployable_models/apple/A_cd5k_from2000/brut")
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"[infer] modele = {MODEL}")
print(f"[infer] device = {DEVICE}  | torch {torch.__version__}")

# 1) policy : config + poids charges directement depuis le checkpoint (sans dataset)
policy = DiffusionPolicy.from_pretrained(str(MODEL))
policy.eval().to(DEVICE)
policy.reset()

# 2) preprocessor / postprocessor (normalisation sauvee DANS le checkpoint)
#    override device_processor : le checkpoint a ete entraine sur mps -> forcer le device local
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(MODEL),
    preprocessor_overrides={"device_processor": {"device": DEVICE.type}})

# 3) features attendues -> construit une observation factice coherente
inp = policy.config.input_features
print(f"[infer] input_features attendues : { {k: tuple(v.shape) for k,v in inp.items()} }")
print(f"[infer] output_features : { {k: tuple(v.shape) for k,v in policy.config.output_features.items()} }")

obs = {}
for key, ft in inp.items():
    shape = tuple(ft.shape)
    if "image" in key:                       # [3,224,224] float [0,1]
        obs[key] = torch.rand(1, *shape, dtype=torch.float32)
    else:                                     # observation.state [5] rad
        obs[key] = torch.rand(1, *shape, dtype=torch.float32)

# 4) inference : preprocessor -> select_action (chunking gere en interne) -> postprocessor
with torch.no_grad():
    action = policy.select_action(preprocessor(obs))
    action = postprocessor(action)

a = action.squeeze(0).cpu().numpy()
print(f"[infer] ACTION produite (dim {a.shape[-1]}) = {np.round(a, 4)}")
print(f"[infer]   -> 5 premiers = consignes articulaires (rad), dernier = gripper")
print("[infer] OK — le modele se charge et infere.")
