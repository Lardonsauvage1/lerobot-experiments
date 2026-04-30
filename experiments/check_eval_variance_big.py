"""10 évals de 200 épisodes (seeds différents) sur le modèle 28_baseline."""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path

MODEL_DIR = Path("results/runs/28_baseline_no_hist_ep10000")
IMAGE_SIZE = 64
MAX_STEPS = 300
N_BATCHES = 10
N_EPS_PER_BATCH = 200


class MLPChunk(nn.Module):
    def __init__(self, feature_dim, hidden_layers, chunk_size,
                 pos_dim=2, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        layers = []
        prev = feature_dim + pos_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

    def forward(self, features, agent_pos):
        x = torch.cat([features, agent_pos], dim=1)
        out = self.net(x)
        return out.reshape(-1, self.chunk_size, 2)


def load_model():
    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    model = MLPChunk(cfg["feature_dim"], cfg["hidden_layers"], cfg["chunk_size"])
    state = torch.load(MODEL_DIR / "model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_norm():
    cache = torch.load("data_cache/resnet18_features_ep206_64px.pt", weights_only=True)
    return cache["norm"]


def eval_run(model, backbone, norm, n_episodes, seed_offset):
    import gymnasium as gym
    import gym_pusht  # noqa
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    coverages = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=(seed_offset + ep))
        max_coverage = 0
        action_buffer = []
        for step in range(MAX_STEPS):
            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")
                with torch.no_grad():
                    features = backbone(img).flatten(1)
                    chunk_pred = model(features, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)
            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, _, _, info = env.step(action)
            max_coverage = max(max_coverage, info.get("coverage", 0))
        coverages.append(max_coverage)
    env.close()
    return coverages


def main():
    print(f"Modèle : {MODEL_DIR}")
    model = load_model()
    norm = get_norm()
    from torchvision.models import resnet18, ResNet18_Weights
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    print(f"\n{N_BATCHES} batches × {N_EPS_PER_BATCH} épisodes (seeds différents)")
    print(f"  Total : {N_BATCHES * N_EPS_PER_BATCH} épisodes")
    print(f"  Estimé : ~{N_BATCHES * N_EPS_PER_BATCH * 0.8 / 60:.0f} min\n")

    means = []
    t_total = time.time()
    for b in range(N_BATCHES):
        # Seeds non-chevauchants entre batches
        seed_offset = 100000 + b * N_EPS_PER_BATCH
        t0 = time.time()
        covs = eval_run(model, backbone, norm, N_EPS_PER_BATCH, seed_offset)
        elapsed = time.time() - t0
        mean_cov = np.mean(covs) * 100
        means.append(mean_cov)
        print(f"  Batch {b+1:2d}/10 (seeds {seed_offset}-{seed_offset+N_EPS_PER_BATCH-1}) : "
              f"coverage = {mean_cov:.2f}%  ({elapsed:.0f}s)")

    total = time.time() - t_total
    print(f"\nTemps total : {total/60:.1f} min")

    print("\n" + "=" * 60)
    print("LES 10 VALEURS")
    print("=" * 60)
    for i, m in enumerate(means):
        print(f"  {i+1:2d} : {m:.2f}%")

    print()
    print(f"  Moyenne     : {np.mean(means):.2f}%")
    print(f"  Écart-type  : {np.std(means):.2f}pt")
    print(f"  Min - Max   : {min(means):.2f}% - {max(means):.2f}% (spread {max(means)-min(means):.2f}pt)")


if __name__ == "__main__":
    main()
