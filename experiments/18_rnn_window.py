"""
Expérience 18 — RNN avec fenêtre glissante + action chunking.

Le RNN reçoit les N dernières frames en séquence, puis prédit un chunk de 20 actions.
On teste différentes tailles de fenêtre.

Architecture :
    [Frame t-N, ..., Frame t-1, Frame t] → chacune passe dans le CNN
    → séquence de features → GRU → état caché final → Linear → chunk 20 actions
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import shutil
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.data_cache import load_pusht_cached
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time,
    save_checkpoint
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "18_rnn_window"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

RNN_HIDDEN = 128
RNN_LAYERS = 1
CHUNK_SIZE = 20
WINDOW_SIZE = 5

LEARNING_RATE = 1e-3
EPOCHS = 300
BATCH_SIZE = 32
SEED = 42

N_EPISODES_TRAIN = 206
N_EVAL_EPISODES = 50
MAX_STEPS = 300
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


class RNNWindowChunkPolicy(nn.Module):
    """CNN + GRU avec fenêtre glissante + action chunking.

    Entraînement :
        [img_t-N, ..., img_t] → CNN (chaque frame) → séquence de features
        [pos_t-N, ..., pos_t] → concaténer avec features
        séquence → GRU → état caché final → Linear → 20 actions

    Inférence :
        On maintient un buffer des N dernières frames.
        À chaque prédiction de chunk, on passe le buffer entier dans le RNN.
    """

    def __init__(self, cnn_feature_dim, image_size, rnn_hidden, rnn_layers,
                 chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        input_dim = cnn_feature_dim + pos_dim
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=rnn_hidden,
            num_layers=rnn_layers, batch_first=True,
        )
        self.fc_out = nn.Linear(rnn_hidden, action_dim * chunk_size)

    def forward(self, images_seq, pos_seq):
        """
        images_seq : (batch, window, 3, H, W)
        pos_seq    : (batch, window, 2)
        Retourne   : (batch, chunk_size, action_dim)
        """
        batch, window = images_seq.shape[:2]

        # CNN sur toutes les frames
        imgs_flat = images_seq.reshape(batch * window, *images_seq.shape[2:])
        feats = self.cnn(imgs_flat).reshape(batch, window, -1)

        # Concaténer features + positions
        x = torch.cat([feats, pos_seq], dim=2)  # (batch, window, input_dim)

        # GRU
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]  # (batch, rnn_hidden)

        # Prédire le chunk
        out = self.fc_out(last_hidden)
        return out.reshape(-1, self.chunk_size, self.action_dim)


def load_data_windowed():
    """Charge les données avec fenêtre glissante + chunk."""
    print(f"Chargement (window={WINDOW_SIZE}, chunk={CHUNK_SIZE})...")

    X_img, X_pos, Y_single, norm, meta = load_pusht_cached(
        n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET, episodes=list(range(N_EPISODES_TRAIN)))
    all_eps = [dataset[i]["episode_index"].item() for i in range(len(dataset))]

    # On a besoin de WINDOW_SIZE frames avant + CHUNK_SIZE frames après
    total_needed = WINDOW_SIZE + CHUNK_SIZE - 1

    seq_imgs, seq_pos, seq_actions = [], [], []
    i = 0
    while i + total_needed <= len(X_img):
        # Vérifier que tout est dans le même épisode
        if all_eps[i] == all_eps[i + total_needed - 1]:
            # Fenêtre : frames [i, i+WINDOW_SIZE)
            seq_imgs.append(X_img[i:i + WINDOW_SIZE])
            seq_pos.append(X_pos[i:i + WINDOW_SIZE])
            # Chunk : actions [i+WINDOW_SIZE-1, i+WINDOW_SIZE-1+CHUNK_SIZE)
            chunk_start = i + WINDOW_SIZE - 1
            seq_actions.append(Y_single[chunk_start:chunk_start + CHUNK_SIZE])
            i += 1
        else:
            i += 1

    X_img_w = torch.stack(seq_imgs)    # (N, window, 3, H, W)
    X_pos_w = torch.stack(seq_pos)     # (N, window, 2)
    Y_w = torch.stack(seq_actions)     # (N, chunk_size, 2)

    print(f"  {len(X_img_w)} samples (window={WINDOW_SIZE}, chunk={CHUNK_SIZE})")
    return X_img_w, X_pos_w, Y_w, norm


def train_model(model, X_img, X_pos, Y, norm_params):
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

    run_dir = get_run_dir(EXPERIMENT_NAME, f"window{WINDOW_SIZE}")

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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s")

        if (epoch + 1) % 100 == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, test_losses, epoch_times, run_dir)
            shutil.copy(run_dir / "checkpoint.pt", run_dir / f"checkpoint_ep{epoch+1}.pt")
            print(f"  Checkpoint (epoch {epoch+1})")
            # Éval intermédiaire
            print(f"  Éval intermédiaire (epoch {epoch+1})...")
            eval_res, _ = eval_in_simulation(model, norm_params)
            print(f"  → Coverage: {eval_res['avg_coverage']:.1%}, Success: {eval_res['success_rate']:.0%}")
            model.train()  # remettre en mode entraînement

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s")
    return train_losses, test_losses, epoch_times, total_time, len(train_idx), len(test_idx)


def eval_in_simulation(model, norm):
    import gymnasium as gym
    import gym_pusht  # noqa
    from collections import deque

    print(f"\n  Évaluation simulation ({N_EVAL_EPISODES} épisodes, window={WINDOW_SIZE})...")
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

        # Buffer pour la fenêtre glissante
        img_buffer = deque(maxlen=WINDOW_SIZE)
        pos_buffer = deque(maxlen=WINDOW_SIZE)

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            # Ajouter l'observation au buffer
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            agent_pos_norm = (agent_pos - norm["p_mean"]) / norm["p_std"]
            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear").squeeze(0)

            img_buffer.append(img)
            pos_buffer.append(agent_pos_norm)

            if len(action_buffer) == 0:
                # Remplir le buffer si pas assez de frames (début d'épisode)
                while len(img_buffer) < WINDOW_SIZE:
                    img_buffer.appendleft(img_buffer[0])
                    pos_buffer.appendleft(pos_buffer[0])

                # Prédire
                imgs = torch.stack(list(img_buffer)).unsqueeze(0)   # (1, window, 3, H, W)
                poss = torch.stack(list(pos_buffer)).unsqueeze(0)   # (1, window, 2)

                with torch.no_grad():
                    chunk_pred = model(imgs, poss)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        status = "✓" if success else "✗"
        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1:2d}/{N_EVAL_EPISODES} : {status} coverage={max_coverage:.1%}")
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
    label = f"window{WINDOW_SIZE}"
    arch_name = f"CNN+GRU(h={RNN_HIDDEN}) window={WINDOW_SIZE} chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    X_img, X_pos, Y, norm = load_data_windowed()

    # 2. Modèle
    model = RNNWindowChunkPolicy(
        cnn_feature_dim=CNN_FEATURE_DIM, image_size=IMAGE_SIZE,
        rnn_hidden=RNN_HIDDEN, rnn_layers=RNN_LAYERS,
        chunk_size=CHUNK_SIZE,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres : {info['total_params']:,}")

    # 3. FLOPs
    dummy_imgs = torch.randn(1, WINDOW_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_pos = torch.randn(1, WINDOW_SIZE, 2)
    flops_inf = count_flops(model, {"images_seq": dummy_imgs, "joints_seq": dummy_pos})
    print(f"  FLOPs/appel  : {flops_inf:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y, norm)

    # 5. Inférence
    inf_time = measure_inference_time(model, {"images_seq": dummy_imgs, "joints_seq": dummy_pos})
    print(f"  Inférence : {inf_time:.2f}ms/appel ({inf_time/CHUNK_SIZE:.2f}ms/frame)")

    # 6. Évaluation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 7. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "RNNWindowChunkPolicy",
        "cnn_feature_dim": CNN_FEATURE_DIM, "image_size": IMAGE_SIZE,
        "rnn_hidden": RNN_HIDDEN, "rnn_layers": RNN_LAYERS,
        "chunk_size": CHUNK_SIZE, "window_size": WINDOW_SIZE,
    }
    save_model(model, run_dir, model_config)

    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    batches_per_epoch = n_train // BATCH_SIZE + 1
    flops_total_train = flops_inf * BATCH_SIZE * batches_per_epoch * EPOCHS

    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture=arch_name,
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
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "window_size": WINDOW_SIZE, "rnn_hidden": RNN_HIDDEN,
            "chunk_size": CHUNK_SIZE,
            "description": f"RNN fenêtre glissante={WINDOW_SIZE} + chunk={CHUNK_SIZE}",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — window={WINDOW_SIZE}, chunk={CHUNK_SIZE}")
    print(f"{'='*60}")
    print(f"  Loss test     : {test_losses[-1]:.6f}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Params        : {info['total_params']:,}")
    print(f"  Inférence     : {inf_time:.2f}ms/appel ({inf_time/CHUNK_SIZE:.2f}ms/frame)")
    print(f"  Temps train   : {train_time:.0f}s")
    print(f"  Dossier       : {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    WINDOW_SIZE = args.window
    main()
