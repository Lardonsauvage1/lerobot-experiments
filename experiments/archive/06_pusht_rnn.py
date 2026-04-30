"""
Expérience 06 — RNN sur PushT avec évaluation en simulation.

Le MLP (exp 05) ne marchait pas : 0% de succès car il ne voit qu'un instant.
Le RNN a une mémoire — il voit la séquence et peut planifier un mouvement.

On compare :
  Run 1 : CNN + RNN, séquence de 5 frames
  Run 2 : CNN + RNN, séquence de 10 frames
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

EXPERIMENT_NAME = "06_pusht_rnn"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64
RNN_HIDDEN = 64

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
NORMALIZE = True

SEQ_LEN = 5
N_EPISODES_TRAIN = 50       # Nombre d'épisodes pour l'entraînement (None = tout)
N_EVAL_EPISODES = 10
MAX_STEPS = 300

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


class CNNRNNPolicy(nn.Module):
    def __init__(self, cnn_channels, cnn_feature_dim, image_size, rnn_hidden, pos_dim=2, output_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(cnn_channels, cnn_feature_dim, image_size)
        input_dim = cnn_feature_dim + pos_dim
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden, batch_first=True)
        self.fc_out = nn.Linear(rnn_hidden, output_dim)
        self.rnn_hidden = rnn_hidden

    def forward(self, images_seq, pos_seq):
        # images_seq: (batch, seq_len, 3, H, W)
        # pos_seq: (batch, seq_len, 2)
        batch, seq_len = images_seq.shape[:2]
        imgs_flat = images_seq.reshape(batch * seq_len, *images_seq.shape[2:])
        feats = self.cnn(imgs_flat).reshape(batch, seq_len, -1)
        x = torch.cat([feats, pos_seq], dim=2)
        rnn_out, _ = self.rnn(x)
        return self.fc_out(rnn_out[:, -1, :])

    def predict_step(self, image, pos, hidden):
        """Prédit une action à partir d'une seule frame + état caché (pour l'éval)."""
        feats = self.cnn(image.unsqueeze(0))
        x = torch.cat([feats, pos.unsqueeze(0)], dim=1).unsqueeze(1)  # (1, 1, input_dim)
        rnn_out, hidden = self.rnn(x, hidden)
        action = self.fc_out(rnn_out[:, -1, :])
        return action, hidden


def load_data_sequences():
    print(f"Chargement du dataset {DATASET}...")
    episodes = list(range(N_EPISODES_TRAIN)) if N_EPISODES_TRAIN else None
    dataset = LeRobotDataset(DATASET, episodes=episodes)
    print(f"  Frames   : {len(dataset)}")
    print(f"  Épisodes : {dataset.num_episodes}")

    all_imgs, all_pos, all_actions, all_eps = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        img = F.interpolate(s["observation.image"].unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)
        all_imgs.append(img)
        all_pos.append(s["observation.state"])
        all_actions.append(s["action"])
        all_eps.append(s["episode_index"].item())

    all_imgs = torch.stack(all_imgs).float()
    all_pos = torch.stack(all_pos).float()
    all_actions = torch.stack(all_actions).float()

    # Normaliser
    p_mean, p_std = all_pos.mean(0), all_pos.std(0)
    y_mean, y_std = all_actions.mean(0), all_actions.std(0)
    p_std = torch.where(p_std > 1e-6, p_std, torch.ones_like(p_std))
    y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
    all_pos = (all_pos - p_mean) / p_std
    all_actions = (all_actions - y_mean) / y_std

    norm = {"p_mean": p_mean, "p_std": p_std, "y_mean": y_mean, "y_std": y_std}

    # Découper en séquences
    seq_imgs, seq_pos, seq_actions = [], [], []
    i = 0
    while i + SEQ_LEN <= len(all_imgs):
        if all_eps[i] == all_eps[i + SEQ_LEN - 1]:
            seq_imgs.append(all_imgs[i:i + SEQ_LEN])
            seq_pos.append(all_pos[i:i + SEQ_LEN])
            seq_actions.append(all_actions[i + SEQ_LEN - 1])
            i += 1
        else:
            i += 1

    X_img = torch.stack(seq_imgs)
    X_pos = torch.stack(seq_pos)
    Y = torch.stack(seq_actions)

    print(f"  Séquences (len={SEQ_LEN}) : {len(X_img)}")
    return X_img, X_pos, Y, norm


def train_model(model, X_img, X_pos, Y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_img)
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
            pred = model(X_img[batch], X_pos[batch])
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            test_pred = model(X_img[test_idx], X_pos[test_idx])
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
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n{'='*60}")
    print(f"ÉVALUATION EN SIMULATION ({N_EVAL_EPISODES} épisodes, seq={SEQ_LEN})")
    print(f"{'='*60}")

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
        hidden = None

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            # Préparer observation
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            agent_pos_norm = (agent_pos - norm["p_mean"]) / norm["p_std"]

            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)

            # Prédire avec le RNN (frame par frame avec état caché)
            with torch.no_grad():
                action_pred, hidden = model.predict_step(img, agent_pos_norm, hidden)

            action = (action_pred * norm["y_std"] + norm["y_mean"]).squeeze(0).numpy()
            action = np.clip(action, 0, 512).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            coverage = info.get("coverage", 0)
            max_coverage = max(max_coverage, coverage)

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        status = "✓" if success else "✗"
        print(f"  Episode {ep+1:2d} : {status} reward={ep_reward:.2f}, coverage={max_coverage:.1%}")

        if frames:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])

    print(f"\n  Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"  Reward moyen : {avg_reward:.2f}")
    print(f"  Coverage moyen : {avg_coverage:.1%}")

    # Sauvegarder frames clés
    if all_frames:
        os.makedirs("results/videos", exist_ok=True)
        for ep_idx, frames in enumerate(all_frames):
            key_frames = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, -1]
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, fi in enumerate(key_frames):
                axes[i].imshow(frames[fi])
                axes[i].set_title(f"Step {fi if fi >= 0 else len(frames)-1}")
                axes[i].axis("off")
            plt.suptitle(f"Episode {ep_idx+1} — RNN seq={SEQ_LEN}")
            plt.tight_layout()
            path = f"results/videos/{EXPERIMENT_NAME}_seq{SEQ_LEN}_ep{ep_idx}.png"
            plt.savefig(path, dpi=100)
            plt.close()
            print(f"  Frames : {path}")

    return {"success_rate": n_success / N_EVAL_EPISODES,
            "avg_reward": avg_reward, "avg_coverage": avg_coverage}


def main():
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} (seq={SEQ_LEN})")
    print(f"{'='*60}")

    X_img, X_pos, Y, norm = load_data_sequences()

    model = CNNRNNPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, rnn_hidden=RNN_HIDDEN,
    )
    info = model_info(model, name=f"CNN+RNN seq={SEQ_LEN}")
    print(f"\n  Paramètres : {info['total_params']:,}")
    print(f"  Taille     : {info['size_mb']:.2f} Mo")
    print()

    print("Entraînement...")
    train_losses, test_losses, train_time = train_model(model, X_img, X_pos, Y)

    eval_results = eval_in_simulation(model, norm)

    # Graphe
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train", linewidth=2)
    ax.plot(test_losses, label="Test", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Loss — PushT RNN seq={SEQ_LEN} (success={eval_results['success_rate']:.0%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{EXPERIMENT_NAME}_seq{SEQ_LEN}_loss.png", dpi=150)
    plt.close()

    config = {
        "seq_len": SEQ_LEN, "rnn_hidden": RNN_HIDDEN,
        "epochs": EPOCHS, "n_params": info["total_params"],
    }
    metrics = {
        "final_train_loss": train_losses[-1], "final_test_loss": test_losses[-1],
        "success_rate": eval_results["success_rate"],
        "avg_coverage": eval_results["avg_coverage"],
        "avg_reward": eval_results["avg_reward"],
        "train_time_s": train_time,
    }
    save_run(EXPERIMENT_NAME, config, metrics)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — RNN seq={SEQ_LEN}")
    print(f"{'='*60}")
    print(f"  Loss test     : {test_losses[-1]:.6f}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    SEQ_LEN = args.seq_len
    N_EPISODES_TRAIN = args.episodes
    main()
