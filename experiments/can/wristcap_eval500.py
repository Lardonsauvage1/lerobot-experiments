#!/usr/bin/env python
"""Éval 500 rollouts WRISTCAP — UNIQUEMENT les 2 bras A (agentview) & B (agentview+poignet).
Réutilise le moteur de rollout de 10_vision_500_rollouts.py (mêmes 500 états figés, IC95 Wilson).
Sortie : results/runs/can/wristcap_eval500.json. À lancer sur Principal (robosuite)."""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)   # charge robosuite + helpers (make_eval_states, rollout_can, main)

m.MODELS = [
    {"name": "wristcap_A_agentview_hicap (cd)",
     "ckpt": "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model",
     "cams": ["agentview"], "image_keys": {"agentview": "observation.image"}},
    {"name": "wristcap_B_wrist_hicap (cd)",
     "ckpt": "results/runs/can/wristcap_B_wrist/cooldown/checkpoints/005000/pretrained_model",
     "cams": ["agentview", "robot0_eye_in_hand"],
     "image_keys": {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}},
]
m.OUT = Path("results/runs/can/wristcap_eval500.json")
m.main()
