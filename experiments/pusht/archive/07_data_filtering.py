"""
Expérience 07 — Data filtering sur PushT.

On filtre le dataset pour ne garder que les frames où reward > 0
(l'agent est en train de progresser vers le but).

Runs :
  1. Avec filtre (reward > 0)
  2. Sans filtre (baseline, même conditions)
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.tracker import build_run_data, save_run, count_flops, measure_inference_time
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "07_data_filtering"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64
RNN_HIDDEN = 64

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

SEQ_LEN = 5
N_EPISODES_TRAIN = 50
N_EVAL_EPISODES = 10
MAX_STEPS = 300

FILTER_REWARD = False  # True = garder seulement reward > 0

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

    def forward(self, images_seq, pos_seq):
        batch, seq_len = images_seq.shape[:2]
        imgs_flat = images_seq.reshape(batch * seq_len, *images_seq.shape[2:])
        feats = self.cnn(imgs_flat).reshape(batch, seq_len, -1)
        x = torch.cat([feats, pos_seq], dim=2)
        rnn_out, _ = self.rnn(x)
        return self.fc_out(rnn_out[:, -1, :])

    def predict_step(self, image, pos, hidden):
        feats = self.cnn(image.unsqueeze(0))
        x = torch.cat([feats, pos.unsqueeze(0)], dim=1).unsqueeze(1)
        rnn_out, hidden = self.rnn(x, hidden)
        action = self.fc_out(rnn_out[:, -1, :])
        return action, hidden


def load_data_sequences():
    print(f"Chargement du dataset {DATASET} ({N_EPISODES_TRAIN} épisodes)...")
    episodes = list(range(N_EPISODES_TRAIN))
    dataset = LeRobotDataset(DATASET, episodes=episodes)
    print(f"  Frames brutes : {len(dataset)}")

    all_imgs, all_pos, all_actions, all_eps, all_rewards = [], [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        img = F.interpolate(s["observation.image"].unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)
        all_imgs.append(img)
        all_pos.append(s["observation.state"])
        all_actions.append(s["action"])
        all_eps.append(s["episode_index"].item())
        all_rewards.append(s["next.reward"].item())

    # Filtrer si demandé
    if FILTER_REWARD:
        keep = [i for i, r in enumerate(all_rewards) if r > 0]
        print(f"  Filtre reward > 0 : {len(keep)}/{len(all_imgs)} frames gardées ({len(keep)/len(all_imgs):.0%})")
        all_imgs = [all_imgs[i] for i in keep]
        all_pos = [all_pos[i] for i in keep]
        all_actions = [all_actions[i] for i in keep]
        all_eps = [all_eps[i] for i in keep]
    else:
        print(f"  Pas de filtre, toutes les frames gardées")

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
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
        hidden = None

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            agent_pos_norm = (agent_pos - norm["p_mean"]) / norm["p_std"]
            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)

            with torch.no_grad():
                action_pred, hidden = model.predict_step(img, agent_pos_norm, hidden)

            action = (action_pred * norm["y_std"] + norm["y_mean"]).squeeze(0).numpy()
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

    # Frames clés
    if all_frames:
        os.makedirs("results/videos", exist_ok=True)
        label = "filtered" if FILTER_REWARD else "unfiltered"
        for ep_idx, frames in enumerate(all_frames):
            key_frames = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, -1]
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, fi in enumerate(key_frames):
                axes[i].imshow(frames[fi])
                axes[i].set_title(f"Step {fi if fi >= 0 else len(frames)-1}")
                axes[i].axis("off")
            plt.suptitle(f"Episode {ep_idx+1} — RNN {label}")
            plt.tight_layout()
            path = f"results/videos/{EXPERIMENT_NAME}_{label}_ep{ep_idx}.png"
            plt.savefig(path, dpi=100)
            plt.close()

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
    }


def main():
    label = "filtré (reward>0)" if FILTER_REWARD else "non filtré"
    arch = f"CNN+RNN seq={SEQ_LEN}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    X_img, X_pos, Y, norm = load_data_sequences()

    # 2. Modèle
    model = CNNRNNPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, rnn_hidden=RNN_HIDDEN,
    )
    info = model_info(model, name=arch)
    print(f"  Paramètres : {info['total_params']:,}")

    # 3. Mesurer FLOPs
    dummy = {
        "images_seq": torch.randn(1, SEQ_LEN, 3, IMAGE_SIZE, IMAGE_SIZE),
        "joints_seq": torch.randn(1, SEQ_LEN, 2),
    }
    flops_inf = count_flops(model, dummy)
    print(f"  FLOPs/inférence : {flops_inf:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y)

    # 5. Temps d'inférence
    inf_time = measure_inference_time(model, {
        "images_seq": torch.randn(1, SEQ_LEN, 3, IMAGE_SIZE, IMAGE_SIZE),
        "joints_seq": torch.randn(1, SEQ_LEN, 2),
    })
    print(f"  Inférence : {inf_time:.2f}ms (par frame : {inf_time/SEQ_LEN:.2f}ms)")

    # 6. Évaluation simulation
    eval_results = eval_in_simulation(model, norm)

    # 7. FLOPs totaux entraînement
    batches_per_epoch = n_train // BATCH_SIZE + 1
    flops_total_train = flops_inf * BATCH_SIZE * batches_per_epoch * EPOCHS

    # 8. Sauvegarder avec le nouveau tracker
    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture=f"{arch} {'filtered' if FILTER_REWARD else 'unfiltered'}",
        dataset=DATASET,
        n_episodes=N_EPISODES_TRAIN,
        n_params=info["total_params"],
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
        extra={"filter": "reward>0" if FILTER_REWARD else "none", "seq_len": SEQ_LEN},
    )
    save_run(run_data)

    # 9. Graphes
    plot_run_losses(run_data)

    # 10. Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — {label}")
    print(f"{'='*60}")
    print(f"  Loss test     : {test_losses[-1]:.6f}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Inférence     : {inf_time:.2f}ms ({inf_time/SEQ_LEN:.2f}ms/frame)")
    print(f"  FLOPs/inf     : {flops_inf:,}")
    print(f"  Temps train   : {train_time:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action="store_true", help="Filtrer reward > 0")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    FILTER_REWARD = args.filter
    SEQ_LEN = args.seq_len
    N_EPISODES_TRAIN = args.episodes
    main()
