#!/usr/bin/env python
"""Génère pusht_hilserl.json (TrainRLServerPipelineConfig) pour HIL-SERL image sur Push-T."""
import json
import draccus
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig, HILSerlProcessorConfig
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.configs.default import DatasetConfig

IMG = 96
OUT = "experiments/hilserl_pusht/pusht_hilserl.json"

env = HILSerlRobotEnvConfig(
    task="PushT-v0", fps=10, name="gym_hil", robot=None, teleop=None,
    features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(IMG, IMG, 3)),
    },
    features_map={"action": "action", "agent_pos": "observation.state", "pixels": "observation.image"},
    processor=HILSerlProcessorConfig(reward_classifier=None, gripper=None, reset=None),
)

policy = SACConfig(
    device="cpu", storage_device="cpu",   # MPS instable sur ce M1 (cholesky MultivariateNormal, CNN) -> CPU fiable, Push-T assez léger
    vision_encoder_name="helper2424/resnet10", freeze_vision_encoder=True,   # ResNet-10 pré-entraîné ImageNet (comme HIL-SERL), gelé = extracteur de features + rapide CPU
    num_discrete_actions=None, use_torch_compile=False,
    push_to_hub=False,
    online_step_before_learning=500,
    # --- RLPD (cœur de HIL-SERL) : UTD élevé + ensemble de critiques -> le critique suit l'acteur ---
    utd_ratio=10,                 # ~20 dans RLPD ; 10 = compromis Mac (défaut 1 = FAUX, cause l'érosion)
    num_critics=10,               # ensemble RLPD (défaut 2)
    num_subsample_critics=2,      # sous-échantillonne 2 critiques pour la cible Bellman (anti-surestimation)
    dataset_stats={
        "observation.image": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "observation.state": {"min": [0.0, 0.0], "max": [512.0, 512.0]},
        "action": {"min": [0.0, 0.0], "max": [512.0, 512.0]},
    },
)

cfg = TrainRLServerPipelineConfig(
    env=env, policy=policy,
    dataset=DatasetConfig(repo_id="local/pusht_sparse", root="data_cache/lerobot_pusht_sparse"),  # reward SPARSE (succès seulement)
    output_dir="results/runs/hilserl_pusht/run1", job_name="pusht_hilserl",
    batch_size=128, steps=100000, seed=1, save_freq=5000, log_freq=200,
)

with open(OUT, "w") as f:
    json.dump(draccus.encode(cfg), f, indent=2)   # enums -> chaînes, JSON propre
print(f"-> {OUT} généré")
