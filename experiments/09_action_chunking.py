"""
Expérience 09 — Action Chunking sur PushT.

Au lieu de prédire 1 action, le modèle prédit N actions futures d'un coup.
On exécute le chunk entier sans rappeler le modèle, puis on re-prédit.

Runs :
  1. chunk=1 (baseline, frame par frame)
  2. chunk=5
  3. chunk=10
  4. chunk=20
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")

from src.data_cache import load_pusht_cached
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "09_action_chunking"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64
HIDDEN_LAYERS = [128, 64]

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 50
N_EVAL_EPISODES = 10
MAX_STEPS = 300

CHUNK_SIZE = 1  # Nombre d'actions futures à prédire

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


class ChunkPolicy(nn.Module):
    """CNN + MLP qui prédit un chunk de N actions futures.

    Architecture :
        Image (3, 64, 64) → CNN → features (64D)
        features + position (2D) → MLP → N actions (2×N valeurs)

    Avec chunk=1 : identique à un MLP classique (2 sorties)
    Avec chunk=10 : prédit 10 actions d'un coup (20 sorties)
    """

    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 hidden_layers, chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        mlp_input = cnn_feature_dim + pos_dim
        output_dim = action_dim * chunk_size

        layers = []
        prev = mlp_input
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, image, agent_pos):
        feats = self.cnn(image)
        x = torch.cat([feats, agent_pos], dim=1)
        out = self.mlp(x)
        # Reshape en (batch, chunk_size, action_dim)
        return out.reshape(-1, self.chunk_size, self.action_dim)


def load_data_chunked():
    """Charge le dataset et crée les cibles chunked.

    Pour chaque frame t, la cible est [action_t, action_t+1, ..., action_t+chunk-1].
    On ne garde que les frames où le chunk complet est dans le même épisode.
    """
    print(f"Chargement (chunk={CHUNK_SIZE})...")

    # Charger les frames individuelles
    X_img, X_pos, Y_single, norm, meta = load_pusht_cached(
        n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
    )

    if CHUNK_SIZE == 1:
        # Pas besoin de chunker, Y est déjà (N, 2)
        Y = Y_single.unsqueeze(1)  # (N, 1, 2)
        print(f"  {len(X_img)} frames, chunk=1")
        return X_img, X_pos, Y, norm

    # Reconstruire les épisodes pour chunker
    # On recharge le dataset pour avoir les episode_index
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    episodes_list = list(range(N_EPISODES_TRAIN))
    dataset = LeRobotDataset(DATASET, episodes=episodes_list)

    all_eps = []
    for i in range(len(dataset)):
        all_eps.append(dataset[i]["episode_index"].item())

    # Créer les chunks
    valid_imgs, valid_pos, valid_chunks = [], [], []
    i = 0
    while i + CHUNK_SIZE <= len(X_img):
        # Vérifier que tout le chunk est dans le même épisode
        if all_eps[i] == all_eps[i + CHUNK_SIZE - 1]:
            valid_imgs.append(X_img[i])
            valid_pos.append(X_pos[i])
            valid_chunks.append(Y_single[i:i + CHUNK_SIZE])
            i += 1
        else:
            i += 1

    X_img_c = torch.stack(valid_imgs)
    X_pos_c = torch.stack(valid_pos)
    Y_c = torch.stack(valid_chunks)  # (N, chunk_size, 2)

    print(f"  {len(X_img_c)} samples (chunk={CHUNK_SIZE}), perdu {len(X_img) - len(X_img_c)} frames aux frontières")
    return X_img_c, X_pos_c, Y_c, norm


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
            pred = model(X_img[batch], X_pos[batch])  # (batch, chunk, 2)
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

    print(f"\n  Évaluation simulation ({N_EVAL_EPISODES} épisodes, chunk={CHUNK_SIZE})...")
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
        step = 0
        action_buffer = []  # Actions restantes du chunk en cours

        while step < MAX_STEPS:
            if ep < 3:
                frames.append(env.render())

            # Si le buffer est vide, prédire un nouveau chunk
            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                if norm:
                    agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]

                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

                with torch.no_grad():
                    chunk_pred = model(img, agent_pos)  # (1, chunk_size, 2)

                # Dénormaliser toutes les actions du chunk
                if norm:
                    chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                else:
                    chunk_actions = chunk_pred.squeeze(0).numpy()

                action_buffer = list(chunk_actions)

            # Exécuter la prochaine action du buffer
            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))
            step += 1

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
    label = f"chunk{CHUNK_SIZE}"
    arch_name = f"CNN+MLP chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    X_img, X_pos, Y, norm = load_data_chunked()

    # 2. Modèle
    model = ChunkPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, hidden_layers=HIDDEN_LAYERS,
        chunk_size=CHUNK_SIZE,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres : {info['total_params']:,}")
    print(f"  Sortie     : {CHUNK_SIZE} actions × 2 = {CHUNK_SIZE * 2} valeurs")

    # 3. FLOPs
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  FLOPs/appel  : {flops_inf:,}")
    print(f"  FLOPs/frame  : {flops_inf // CHUNK_SIZE:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y)

    # 5. Temps d'inférence
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  Inférence : {inf_time:.2f}ms/appel ({inf_time/CHUNK_SIZE:.2f}ms/frame)")

    # 6. Évaluation simulation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 7. Sauvegarder — nouveau format
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    model_config = {
        "class": "ChunkPolicy",
        "cnn_channels": CNN_CHANNELS,
        "cnn_feature_dim": CNN_FEATURE_DIM,
        "image_size": IMAGE_SIZE,
        "hidden_layers": HIDDEN_LAYERS,
        "chunk_size": CHUNK_SIZE,
        "pos_dim": 2,
        "action_dim": 2,
    }
    save_model(model, run_dir, model_config)

    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    # FLOPs totaux entraînement
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
            "chunk_size": CHUNK_SIZE,
            "description": f"Action chunking avec chunk={CHUNK_SIZE} sur PushT",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    # Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — chunk={CHUNK_SIZE}")
    print(f"{'='*60}")
    print(f"  Loss test     : {test_losses[-1]:.6f}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Params        : {info['total_params']:,}")
    print(f"  Inférence     : {inf_time:.2f}ms/appel ({inf_time/CHUNK_SIZE:.2f}ms/frame)")
    print(f"  FLOPs/frame   : {flops_inf // CHUNK_SIZE:,}")
    print(f"  Temps train   : {train_time:.0f}s")
    print(f"  Dossier       : {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=1)
    args = parser.parse_args()
    CHUNK_SIZE = args.chunk
    main()
