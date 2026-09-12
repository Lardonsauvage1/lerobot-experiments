#!/usr/bin/env python
"""Éval 500 rollouts WRISTCAP+224+AUG — bras A' (agentview) & B' (agentview+poignet), rendu 224px.
Réutilise le moteur de 10_vision_500_rollouts.py (mêmes 500 états figés). Sortie wristcap224_eval500.json.
À lancer sur Principal (robosuite)."""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.RENDER_SIZE = 224   # modèles 224px -> rendre les caméras en 224
m.MODELS = [
    {"name": "wristcap224_A_agentview_aug (cd)",
     "ckpt": "results/runs/can/wristcap224_A_agentview/cooldown/checkpoints/005000/pretrained_model",
     "cams": ["agentview"], "image_keys": {"agentview": "observation.images.agentview"}},
    {"name": "wristcap224_B_wrist_aug (cd)",
     "ckpt": "results/runs/can/wristcap224_B_wrist/cooldown/checkpoints/005000/pretrained_model",
     "cams": ["agentview", "robot0_eye_in_hand"],
     "image_keys": {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}},
]
m.OUT = Path("results/runs/can/wristcap224_eval500.json")
m.main()
