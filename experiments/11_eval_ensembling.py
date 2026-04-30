"""
Expérience 11 — Évaluation avec et sans temporal ensembling.

On prend le modèle entraîné (Transformer 2 couches, chunk=20, 206 épisodes)
et on le teste de deux façons :

Run 1 : Normal — 1 appel toutes les 20 frames
Run 2 : Temporal ensembling — 1 appel par frame, moyenne des chunks qui se chevauchent

Le modèle n'est PAS ré-entraîné, seule la façon d'exécuter change.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from collections import deque
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "11_eval_ensembling"
DATASET = "lerobot/pusht"
IMAGE_SIZE = 64
CHUNK_SIZE = 20
SEED = 42

N_EVAL_EPISODES = 10
MAX_STEPS = 300

# Chemin vers le modèle entraîné
MODEL_DIR = None  # sera passé en CLI
USE_ENSEMBLING = False

# ============================================================


class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=64, image_size=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = image_size // 4
        self.fc = nn.Linear(32 * reduced * reduced, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class TransformerChunkPolicy(nn.Module):
    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model
        self.obs_proj = nn.Linear(cnn_feature_dim + pos_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, image, agent_pos):
        batch = image.shape[0]
        cnn_feats = self.cnn(image)
        obs = torch.cat([cnn_feats, agent_pos], dim=1)
        context = self.obs_proj(obs).unsqueeze(1)
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)
        return self.action_head(decoded)


def load_model(model_dir: str):
    """Charge le modèle depuis un dossier de run."""
    model_dir = Path(model_dir)
    config_path = model_dir / "model_config.json"
    weights_path = model_dir / "model.pt"

    with open(config_path) as f:
        config = json.load(f)

    model = TransformerChunkPolicy(
        cnn_channels=config.get("cnn_channels", [16, 32]),
        cnn_feature_dim=config.get("cnn_feature_dim", 64),
        image_size=config.get("image_size", 64),
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 256),
        chunk_size=config.get("chunk_size", 20),
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    print(f"  Modèle chargé depuis {model_dir}")
    return model, config


def load_norm():
    """Charge les paramètres de normalisation depuis le cache."""
    from src.data_cache import load_pusht_cached
    _, _, _, norm, _ = load_pusht_cached(n_episodes=50, image_size=IMAGE_SIZE, seq_len=1)
    return norm


def eval_normal(model, norm):
    """Évaluation normale : 1 appel toutes les CHUNK_SIZE frames."""
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Évaluation NORMALE (1 appel / {CHUNK_SIZE} frames)...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []
    n_model_calls = 0

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        max_coverage = 0
        action_buffer = []

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

                with torch.no_grad():
                    chunk_pred = model(img, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)
                n_model_calls += 1

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        status = "✓" if success else "✗"
        print(f"    Episode {ep+1:2d} : {status} reward={ep_reward:.2f}, coverage={max_coverage:.1%}")
        if frames:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])
    print(f"    Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"    Coverage moyen : {avg_coverage:.1%}")
    print(f"    Appels au modèle : {n_model_calls}")

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
        "n_model_calls": n_model_calls,
    }, all_frames


def eval_ensembling(model, norm):
    """Évaluation avec temporal ensembling : 1 appel par frame, moyenne des chunks."""
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Évaluation ENSEMBLING (1 appel / frame, moyenne des chunks)...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []
    n_model_calls = 0

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        max_coverage = 0

        # Buffer de chunks actifs : chaque élément = (chunk_actions, index_dans_chunk)
        active_chunks = deque()

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            # Prédire un nouveau chunk à chaque frame
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
            agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

            with torch.no_grad():
                chunk_pred = model(img, agent_pos)
            chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
            active_chunks.append(chunk_actions)
            n_model_calls += 1

            # Retirer les chunks expirés (ceux qui n'ont plus d'action pour ce step)
            while len(active_chunks) > CHUNK_SIZE:
                active_chunks.popleft()

            # Moyenne pondérée des actions de tous les chunks actifs
            # Le chunk le plus récent vote pour action[0], le précédent pour action[1], etc.
            action_votes = []
            for i, chunk in enumerate(active_chunks):
                age = len(active_chunks) - 1 - i  # 0 = le plus récent
                if age < len(chunk):
                    action_votes.append(chunk[age])

            if action_votes:
                action = np.mean(action_votes, axis=0)
            else:
                action = np.array([256.0, 256.0])  # centre par défaut

            action = np.clip(action, 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        status = "✓" if success else "✗"
        print(f"    Episode {ep+1:2d} : {status} reward={ep_reward:.2f}, coverage={max_coverage:.1%}")
        if frames:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])
    print(f"    Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"    Coverage moyen : {avg_coverage:.1%}")
    print(f"    Appels au modèle : {n_model_calls}")

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
        "n_model_calls": n_model_calls,
    }, all_frames


def main():
    mode = "ensembling" if USE_ENSEMBLING else "normal"
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {mode}")
    print(f"{'='*60}")
    print(f"  Modèle source : {MODEL_DIR}")
    print(f"  Mode          : {mode}")

    # 1. Charger le modèle
    model, config = load_model(MODEL_DIR)
    info = model_info(model, name=f"Transformer chunk={CHUNK_SIZE} ({mode})")

    # 2. Charger la normalisation
    norm = load_norm()

    # 3. FLOPs et temps d'inférence
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })

    if USE_ENSEMBLING:
        # Ensembling : 1 appel par frame
        frames_per_call = 1
        effective_inf_per_frame = inf_time
    else:
        # Normal : 1 appel par chunk
        frames_per_call = CHUNK_SIZE
        effective_inf_per_frame = inf_time / CHUNK_SIZE

    print(f"  Paramètres     : {info['total_params']:,}")
    print(f"  FLOPs/appel    : {flops_inf:,}")
    print(f"  Inférence/appel: {inf_time:.2f}ms")
    print(f"  Inférence/frame: {effective_inf_per_frame:.2f}ms")

    # 4. Évaluer
    if USE_ENSEMBLING:
        eval_results, all_frames = eval_ensembling(model, norm)
    else:
        eval_results, all_frames = eval_normal(model, norm)

    # 5. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, mode)

    # Copier la config du modèle source
    save_model(model, run_dir, config)

    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture=f"Transformer 2L chunk={CHUNK_SIZE} ({mode})",
        dataset=DATASET,
        n_episodes="206 (modèle pré-entraîné)",
        n_params=info["total_params"],
        size_mb=info["size_mb"],
        seed=SEED,
        train_losses=[],
        test_losses=[],
        train_time_s=0,
        epoch_times=[],
        flops_total_train=0,
        n_train_samples=0,
        n_test_samples=0,
        inference_time_ms=inf_time,
        flops_inference=flops_inf,
        frames_per_call=frames_per_call,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "mode": mode,
            "model_source": str(MODEL_DIR),
            "n_model_calls": eval_results["n_model_calls"],
            "description": f"Évaluation {mode} du Transformer 2 couches chunk={CHUNK_SIZE}",
        },
    )
    save_run(run_data)
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — {mode}")
    print(f"{'='*60}")
    print(f"  Success rate     : {eval_results['success_rate']:.0%}")
    print(f"  Coverage         : {eval_results['avg_coverage']:.1%}")
    print(f"  Appels modèle    : {eval_results['n_model_calls']}")
    print(f"  Inférence/frame  : {effective_inf_per_frame:.2f}ms")
    print(f"  Dossier          : {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Chemin vers le dossier du run contenant model.pt et model_config.json")
    parser.add_argument("--ensembling", action="store_true")
    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    USE_ENSEMBLING = args.ensembling
    main()
