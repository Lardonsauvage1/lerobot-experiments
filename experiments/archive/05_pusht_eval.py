"""
Expérience 05 — Entraîner et tester un modèle dans la simulation PushT.

Contrairement aux expériences précédentes, on ne mesure plus seulement la loss :
on teste le modèle dans la simulation et on regarde s'il réussit la tâche.

Dataset : lerobot/pusht
  - observation.state (2D) : position (x, y) de l'agent
  - observation.image (3, 96, 96) : image top-down
  - action (2D) : position cible (x, y)

Simulation : gym_pusht/PushT-v0
  - L'agent doit pousser un T sur une cible
  - Succès = coverage > 90%
  - On sauvegarde des vidéos pour voir le comportement
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.tracker import save_run
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "05_pusht_eval"
DATASET = "lerobot/pusht"

# Modèle
USE_IMAGE = True
IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64
HIDDEN_LAYERS = [128, 64]

# Entraînement
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
NORMALIZE = True

# Évaluation
N_EVAL_EPISODES = 10
MAX_STEPS = 300
SAVE_VIDEOS = True

# ============================================================


class SimpleCNN(nn.Module):
    def __init__(self, channels, feature_dim, image_size):
        super().__init__()
        conv_layers = []
        in_ch = 3
        for out_ch in channels:
            conv_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        reduced = image_size // (2 ** len(channels))
        self.fc = nn.Linear(channels[-1] * reduced * reduced, feature_dim)

    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class PushTPolicy(nn.Module):
    def __init__(self, use_image, image_size, cnn_channels, cnn_feature_dim,
                 hidden_layers, output_dim=2):
        super().__init__()
        self.use_image = use_image

        if use_image:
            self.cnn = SimpleCNN(cnn_channels, cnn_feature_dim, image_size)
            mlp_input = cnn_feature_dim + 2  # features image + agent_pos
        else:
            self.cnn = None
            mlp_input = 2  # agent_pos seulement

        layers = []
        prev = mlp_input
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, image=None, agent_pos=None):
        if self.use_image and image is not None:
            feats = self.cnn(image)
            x = torch.cat([feats, agent_pos], dim=1)
        else:
            x = agent_pos
        return self.mlp(x)


def load_data():
    print(f"Chargement du dataset {DATASET}...")
    dataset = LeRobotDataset(DATASET)
    print(f"  Frames   : {len(dataset)}")
    print(f"  Épisodes : {dataset.num_episodes}")

    images = []
    positions = []
    actions = []

    for i in range(len(dataset)):
        s = dataset[i]
        positions.append(s["observation.state"])
        actions.append(s["action"])
        if USE_IMAGE:
            img = s["observation.image"]
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)
            images.append(img)

    X_pos = torch.stack(positions).float()
    Y = torch.stack(actions).float()
    X_img = torch.stack(images).float() if USE_IMAGE else None

    print(f"  Positions : {X_pos.shape}")
    print(f"  Actions   : {Y.shape}")
    if X_img is not None:
        print(f"  Images    : {X_img.shape}")

    # Normaliser positions et actions
    norm = {}
    if NORMALIZE:
        p_mean, p_std = X_pos.mean(0), X_pos.std(0)
        y_mean, y_std = Y.mean(0), Y.std(0)
        p_std = torch.where(p_std > 1e-6, p_std, torch.ones_like(p_std))
        y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
        X_pos = (X_pos - p_mean) / p_std
        Y = (Y - y_mean) / y_std
        norm = {"p_mean": p_mean, "p_std": p_std, "y_mean": y_mean, "y_std": y_std}

    return X_img, X_pos, Y, norm


def train_model(model, X_img, X_pos, Y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_pos)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            img_b = X_img[batch] if X_img is not None else None
            pred = model(image=img_b, agent_pos=X_pos[batch])
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            img_t = X_img[test_idx] if X_img is not None else None
            test_pred = model(image=img_t, agent_pos=X_pos[test_idx])
            test_loss = criterion(test_pred, Y[test_idx]).item()

        train_losses.append(avg_train)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s")

    total_time = time.time() - start
    print(f"  Temps total : {total_time:.1f}s")
    return train_losses, test_losses, total_time


def eval_in_simulation(model, norm):
    """Teste le modèle dans la simulation PushT et sauvegarde des vidéos."""
    import gymnasium as gym
    import gym_pusht

    print(f"\n{'='*60}")
    print(f"ÉVALUATION EN SIMULATION ({N_EVAL_EPISODES} épisodes)")
    print(f"{'='*60}")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []

    model.eval()

    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        ep_success = False
        max_coverage = 0

        for step in range(MAX_STEPS):
            # Sauvegarder la frame pour la vidéo
            if SAVE_VIDEOS and ep < 3:  # Sauvegarder les 3 premiers épisodes
                frame = env.render()
                frames.append(frame)

            # Préparer l'observation
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
            if NORMALIZE:
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]

            img = None
            if USE_IMAGE:
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

            # Prédire l'action
            with torch.no_grad():
                action_pred = model(image=img, agent_pos=agent_pos)

            # Dénormaliser l'action
            if NORMALIZE:
                action = (action_pred * norm["y_std"] + norm["y_mean"]).squeeze(0).numpy()
            else:
                action = action_pred.squeeze(0).numpy()

            # Clipper dans le range de l'env
            action = np.clip(action, 0, 512)

            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            ep_reward += reward
            coverage = info.get("coverage", 0)
            max_coverage = max(max_coverage, coverage)
            if info.get("is_success", False):
                ep_success = True

        results.append({
            "episode": ep,
            "reward": ep_reward,
            "success": ep_success,
            "max_coverage": max_coverage,
        })

        status = "✓" if ep_success else "✗"
        print(f"  Episode {ep+1:2d} : {status} reward={ep_reward:.2f}, coverage={max_coverage:.1%}")

        if frames:
            all_frames.append(frames)

    env.close()

    # Résumé
    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])

    print(f"\n  Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"  Reward moyen : {avg_reward:.2f}")
    print(f"  Coverage moyen : {avg_coverage:.1%}")

    # Sauvegarder les vidéos
    if SAVE_VIDEOS and all_frames:
        save_videos(all_frames)

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "n_success": n_success,
        "avg_reward": avg_reward,
        "avg_coverage": avg_coverage,
    }


def save_videos(all_frames):
    """Sauvegarde les épisodes en tant que vidéos (séquences d'images → gif)."""
    os.makedirs("results/videos", exist_ok=True)

    for ep_idx, frames in enumerate(all_frames):
        # Sauvegarder en gif avec matplotlib
        fig, ax = plt.subplots(figsize=(3, 3))

        filepath = f"results/videos/{EXPERIMENT_NAME}_ep{ep_idx}.gif"

        # Sauvegarder quelques frames clés en image
        key_frames = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, -1]
        fig_key, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, fi in enumerate(key_frames):
            axes[i].imshow(frames[fi])
            axes[i].set_title(f"Step {fi if fi >= 0 else len(frames)-1}")
            axes[i].axis("off")
        plt.suptitle(f"Episode {ep_idx + 1}")
        plt.tight_layout()
        img_path = f"results/videos/{EXPERIMENT_NAME}_ep{ep_idx}_keyframes.png"
        plt.savefig(img_path, dpi=100)
        plt.close(fig_key)
        plt.close(fig)
        print(f"  Frames clés : {img_path}")


def main():
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"{'='*60}")
    print(f"  Dataset    : {DATASET}")
    print(f"  Use image  : {USE_IMAGE}")
    print(f"  Epochs     : {EPOCHS}")
    print()

    # 1. Charger
    X_img, X_pos, Y, norm = load_data()

    # 2. Créer le modèle
    model = PushTPolicy(
        use_image=USE_IMAGE, image_size=IMAGE_SIZE,
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        hidden_layers=HIDDEN_LAYERS,
    )
    info = model_info(model, name="CNN+MLP PushT")
    print(f"\n  Paramètres : {info['total_params']:,}")
    print(f"  Taille     : {info['size_mb']:.2f} Mo")
    print()

    # 3. Entraîner
    print("Entraînement...")
    train_losses, test_losses, train_time = train_model(model, X_img, X_pos, Y)

    # 4. Tester dans la simulation
    eval_results = eval_in_simulation(model, norm)

    # 5. Graphe de loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train", linewidth=2)
    ax.plot(test_losses, label="Test", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Loss — PushT CNN+MLP (success={eval_results['success_rate']:.0%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{EXPERIMENT_NAME}_loss.png", dpi=150)
    plt.close()

    # 6. Sauvegarder
    config = {
        "dataset": DATASET, "use_image": USE_IMAGE,
        "hidden_layers": str(HIDDEN_LAYERS), "epochs": EPOCHS,
        "n_params": info["total_params"],
    }
    metrics = {
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "success_rate": eval_results["success_rate"],
        "avg_coverage": eval_results["avg_coverage"],
        "avg_reward": eval_results["avg_reward"],
        "train_time_s": train_time,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    save_run(EXPERIMENT_NAME, config, metrics)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  Loss test     : {test_losses[-1]:.6f}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Temps train   : {train_time:.0f}s")
    print(f"  Vidéos dans   : results/videos/")


if __name__ == "__main__":
    main()
