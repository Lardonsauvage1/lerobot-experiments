"""
Expérience 08 — CNN pré-entraîné vs from scratch.

Run 1 : CNN from scratch + MLP (baseline)
Run 2 : ResNet18 pré-entraîné (gelé) + MLP

Les deux avec le nouveau format standardisé :
  - run_info.md, model.pt, model_config.json, losses.png, vidéos MP4
  - FLOPs, temps d'inférence, paliers de performance
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import matplotlib
matplotlib.use("Agg")

from src.data_cache import load_pusht_cached, get_device
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "08_pretrained_cnn"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
HIDDEN_LAYERS = [128, 64]
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 50
N_EVAL_EPISODES = 10
MAX_STEPS = 300

USE_PRETRAINED = False  # True = ResNet18 gelé, False = CNN from scratch

# ============================================================


class SimpleCNN(nn.Module):
    """CNN from scratch : 2 couches conv."""
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


class PretrainedCNN(nn.Module):
    """ResNet18 pré-entraîné sur ImageNet, poids gelés.

    ResNet18 a été entraîné sur des millions d'images (ImageNet).
    Il sait déjà détecter des bords, des formes, des objets.
    On retire la dernière couche (classification) et on garde les features.
    On gèle tous les poids : le CNN ne s'entraîne PAS, seul le MLP apprend.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Garder tout sauf la dernière couche (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (batch, 512, 1, 1)

        # Geler les poids
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection vers la dimension voulue
        self.fc = nn.Linear(512, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x).flatten(1)  # (batch, 512)
        return F.relu(self.fc(feats))


class PushTPolicy(nn.Module):
    def __init__(self, cnn, hidden_layers, pos_dim=2, output_dim=2):
        super().__init__()
        self.cnn = cnn
        mlp_input = cnn.feature_dim + pos_dim

        layers = []
        prev = mlp_input
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, image=None, agent_pos=None):
        feats = self.cnn(image)
        x = torch.cat([feats, agent_pos], dim=1)
        return self.mlp(x)


def train_model(model, X_img, X_pos, Y):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_img)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    epoch_times = []
    start_total = time.time()

    for epoch in range(EPOCHS):
        start_ep = time.time()
        model.train()
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            pred = model(image=X_img[batch], agent_pos=X_pos[batch])
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            test_pred = model(image=X_img[test_idx], agent_pos=X_pos[test_idx])
            test_loss = criterion(test_pred, Y[test_idx]).item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s")

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s")
    return train_losses, test_losses, epoch_times, total_time, len(train_idx), len(test_idx)


def eval_in_simulation(model, norm):
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Évaluation simulation ({N_EVAL_EPISODES} épisodes)...")
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

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
            if norm:
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]

            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

            with torch.no_grad():
                action_pred = model(image=img, agent_pos=agent_pos)

            if norm:
                action = (action_pred * norm["y_std"] + norm["y_mean"]).squeeze(0).numpy()
            else:
                action = action_pred.squeeze(0).numpy()

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

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
    }, all_frames


def main():
    label = "pretrained" if USE_PRETRAINED else "scratch"
    cnn_type = "ResNet18 gelé" if USE_PRETRAINED else "CNN from scratch [16,32]"
    arch_name = f"{cnn_type} + MLP {HIDDEN_LAYERS}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger (depuis le cache)
    print("\nChargement...")
    X_img, X_pos, Y, norm, meta = load_pusht_cached(
        n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
    )

    # 2. Modèle
    if USE_PRETRAINED:
        cnn = PretrainedCNN(feature_dim=64)
    else:
        cnn = SimpleCNN(feature_dim=64, image_size=IMAGE_SIZE)

    model = PushTPolicy(cnn=cnn, hidden_layers=HIDDEN_LAYERS)
    info = model_info(model, name=arch_name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = info["total_params"]
    print(f"  Paramètres total     : {total:,}")
    print(f"  Paramètres entraînés : {trainable:,}")
    if USE_PRETRAINED:
        print(f"  Paramètres gelés     : {total - trainable:,} (ResNet18)")

    # 3. FLOPs
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  FLOPs/inférence      : {flops_inf:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y)

    # 5. Temps d'inférence
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  Inférence : {inf_time:.2f}ms")

    # 6. Évaluation simulation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 7. FLOPs totaux entraînement
    batches_per_epoch = n_train // BATCH_SIZE + 1
    flops_total_train = flops_inf * BATCH_SIZE * batches_per_epoch * EPOCHS

    # 8. Sauvegarder — nouveau format
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    # model_config pour recréer le modèle
    if USE_PRETRAINED:
        model_config = {
            "class": "PushTPolicy",
            "cnn_class": "PretrainedCNN",
            "cnn_feature_dim": 64,
            "hidden_layers": HIDDEN_LAYERS,
            "pos_dim": 2,
            "output_dim": 2,
            "image_size": IMAGE_SIZE,
        }
    else:
        model_config = {
            "class": "PushTPolicy",
            "cnn_class": "SimpleCNN",
            "cnn_feature_dim": 64,
            "cnn_channels": [16, 32],
            "hidden_layers": HIDDEN_LAYERS,
            "pos_dim": 2,
            "output_dim": 2,
            "image_size": IMAGE_SIZE,
        }

    save_model(model, run_dir, model_config)

    # Vidéos MP4
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    # Run data standardisé
    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture=arch_name,
        dataset=DATASET,
        n_episodes=N_EPISODES_TRAIN,
        n_params=total,
        size_mb=info["size_mb"],
        seed=SEED,
        train_losses=train_losses,
        test_losses=test_losses,
        train_time_s=train_time,
        epoch_times=epoch_times,
        flops_total_train=flops_total_train,
        n_train_samples=n_train,
        n_test_samples=n_test,
        inference_time_ms=inf_time,
        flops_inference=flops_inf,
        frames_per_call=1,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "cnn_type": label,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "description": f"CNN {'pré-entraîné (ResNet18 gelé)' if USE_PRETRAINED else 'from scratch'} + MLP sur PushT",
        },
    )
    save_run(run_data)

    # Graphes
    plot_run_losses(run_data, save_dir=str(run_dir))

    # Run info
    generate_run_info(run_data, model, run_dir)

    # Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — {label}")
    print(f"{'='*60}")
    print(f"  Loss test        : {test_losses[-1]:.6f}")
    print(f"  Success rate     : {eval_results['success_rate']:.0%}")
    print(f"  Coverage         : {eval_results['avg_coverage']:.1%}")
    print(f"  Params entraînés : {trainable:,}")
    print(f"  Inférence        : {inf_time:.2f}ms")
    print(f"  FLOPs/inf        : {flops_inf:,}")
    print(f"  Temps train      : {train_time:.0f}s")
    print(f"  Dossier          : {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    USE_PRETRAINED = args.pretrained
    main()
