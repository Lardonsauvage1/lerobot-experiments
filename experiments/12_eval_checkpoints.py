"""
Expérience 12 — Évaluer chaque checkpoint du training 2000 epochs.

Charge chaque checkpoint (ep200, 400, ..., 1400) et l'évalue en simulation.
Crée un dossier complet par checkpoint.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time,
    load_checkpoint
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "12_checkpoints_eval"
DATASET = "lerobot/pusht"
IMAGE_SIZE = 64
CHUNK_SIZE = 20
SEED = 42
N_EVAL_EPISODES = 10
MAX_STEPS = 300

SOURCE_DIR = "results/runs/10_transformer_2layer_ep206_ep2000epochs"
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


def create_model():
    return TransformerChunkPolicy(
        cnn_channels=[16, 32], cnn_feature_dim=64, image_size=64,
        d_model=64, n_heads=4, n_layers=2, dim_feedforward=256,
        chunk_size=CHUNK_SIZE,
    )


def load_norm():
    from src.data_cache import load_pusht_cached
    _, _, _, norm, _ = load_pusht_cached(n_episodes=50, image_size=IMAGE_SIZE, seq_len=1)
    return norm


def eval_in_simulation(model, norm):
    import gymnasium as gym
    import gym_pusht  # noqa

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []
    model.eval()

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

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
    }, all_frames


def eval_checkpoint(checkpoint_path, epoch_num, norm):
    label = f"ep{epoch_num}"
    print(f"\n{'='*60}")
    print(f"Checkpoint epoch {epoch_num}")
    print(f"{'='*60}")

    # Charger le modèle
    model = create_model()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_losses = checkpoint["train_losses"]
    test_losses = checkpoint["test_losses"]
    epoch_times = checkpoint["epoch_times"]

    info = model_info(model, name=f"Transformer 2L chunk={CHUNK_SIZE} ep{epoch_num}")

    # FLOPs et inférence
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })

    # Évaluation simulation
    print(f"  Évaluation simulation...")
    eval_results, all_frames = eval_in_simulation(model, norm)
    print(f"  Coverage : {eval_results['avg_coverage']:.1%}")
    print(f"  Success  : {eval_results['success_rate']:.0%}")

    # Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    model_config = {
        "class": "TransformerChunkPolicy",
        "cnn_channels": [16, 32], "cnn_feature_dim": 64,
        "image_size": 64, "d_model": 64, "n_heads": 4,
        "n_layers": 2, "dim_feedforward": 256, "chunk_size": CHUNK_SIZE,
    }
    save_model(model, run_dir, model_config)

    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    train_time = sum(epoch_times)
    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture=f"Transformer 2L chunk={CHUNK_SIZE}",
        dataset=DATASET,
        n_episodes=206,
        n_params=info["total_params"],
        size_mb=info["size_mb"],
        seed=SEED,
        train_losses=train_losses,
        test_losses=test_losses,
        train_time_s=train_time,
        epoch_times=epoch_times,
        flops_total_train=flops_inf * 64 * (len(train_losses) * 272),  # estimation
        n_train_samples=17389,
        n_test_samples=4347,
        inference_time_ms=inf_time,
        flops_inference=flops_inf,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "epoch": epoch_num,
            "description": f"Transformer 2 couches chunk={CHUNK_SIZE}, évalué à l'epoch {epoch_num}",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    return eval_results["avg_coverage"]


def main():
    print("Chargement de la normalisation...")
    norm = load_norm()

    # Trouver tous les checkpoints
    source = Path(SOURCE_DIR)
    checkpoints = sorted(source.glob("checkpoint_ep*.pt"))
    print(f"Checkpoints trouvés : {len(checkpoints)}")

    results = []
    for cp in checkpoints:
        epoch_num = int(cp.stem.split("ep")[1])
        coverage = eval_checkpoint(str(cp), epoch_num, norm)
        results.append((epoch_num, coverage))

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ — Coverage par epoch")
    print(f"{'='*60}")
    for epoch, coverage in results:
        print(f"  Epoch {epoch:5d} : {coverage:.1%}")


if __name__ == "__main__":
    main()
