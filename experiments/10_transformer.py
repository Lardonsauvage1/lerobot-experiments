"""
Expérience 10 — Transformer vs MLP pour l'action chunking.

On remplace le MLP par un petit Transformer decoder après le CNN.
Baseline = exp 09 chunk=20 (MLP, coverage 16.7%).

Runs :
  1. Transformer 1 couche (4 têtes, dim=64), chunk=20
  2. Transformer 2 couches, chunk=20
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
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

EXPERIMENT_NAME = "10_transformer"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

# Transformer
N_HEADS = 4
D_MODEL = 64
N_LAYERS = 1
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 50
N_EVAL_EPISODES = 10
MAX_STEPS = 300
RESUME_FROM = None

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
    """CNN + Transformer decoder pour prédire un chunk d'actions.

    Architecture :
        Image → CNN → features (64D)
        features + position → projection → context token (1, d_model)

        Transformer decoder :
            - Entrée : chunk_size tokens de query apprenables
            - Context : le context token (cross-attention)
            - Sortie : chunk_size tokens → Linear → actions (chunk_size, 2)

    Les query tokens sont des embeddings apprenables — un par action à prédire.
    Le Transformer utilise l'attention pour que chaque action puisse
    "regarder" le contexte (image + position) et les autres actions.
    """

    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model

        # Projection de l'observation vers d_model
        self.obs_proj = nn.Linear(cnn_feature_dim + pos_dim, d_model)

        # Query tokens apprenables : un par action du chunk
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)

        # Positional encoding pour les queries (pour que le modèle sache
        # que query 0 = première action, query 19 = dernière action)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Projection de sortie
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, image, agent_pos):
        batch = image.shape[0]

        # Encoder l'observation
        cnn_feats = self.cnn(image)
        obs = torch.cat([cnn_feats, agent_pos], dim=1)
        context = self.obs_proj(obs).unsqueeze(1)  # (batch, 1, d_model)

        # Query tokens + positional encoding
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        # (batch, chunk_size, d_model)

        # Transformer decoder : queries attend to context
        decoded = self.transformer_decoder(queries, context)
        # (batch, chunk_size, d_model)

        # Projeter vers les actions
        actions = self.action_head(decoded)
        # (batch, chunk_size, action_dim)

        return actions


def load_data_chunked():
    print(f"Chargement (chunk={CHUNK_SIZE})...")
    X_img, X_pos, Y_single, norm, meta = load_pusht_cached(
        n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET, episodes=list(range(N_EPISODES_TRAIN)))
    all_eps = [dataset[i]["episode_index"].item() for i in range(len(dataset))]

    valid_imgs, valid_pos, valid_chunks = [], [], []
    i = 0
    while i + CHUNK_SIZE <= len(X_img):
        if all_eps[i] == all_eps[i + CHUNK_SIZE - 1]:
            valid_imgs.append(X_img[i])
            valid_pos.append(X_pos[i])
            valid_chunks.append(Y_single[i:i + CHUNK_SIZE])
            i += 1
        else:
            i += 1

    X_img_c = torch.stack(valid_imgs)
    X_pos_c = torch.stack(valid_pos)
    Y_c = torch.stack(valid_chunks)

    print(f"  {len(X_img_c)} samples (chunk={CHUNK_SIZE})")
    return X_img_c, X_pos_c, Y_c, norm


def train_model(model, X_img, X_pos, Y, resume_from=None):
    from src.tracker import save_checkpoint, load_checkpoint

    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_img)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    # Reprendre depuis un checkpoint si demandé
    start_epoch = 0
    train_losses, test_losses = [], []
    epoch_times = []

    if resume_from and Path(resume_from).exists():
        start_epoch, train_losses, test_losses, epoch_times = load_checkpoint(
            Path(resume_from), model, optimizer
        )
        start_epoch += 1  # reprendre à l'epoch suivante
        print(f"  Reprise à l'epoch {start_epoch + 1}/{EPOCHS}")

    # Dossier pour les checkpoints
    run_dir = get_run_dir(EXPERIMENT_NAME,
                          f"{N_LAYERS}layer_ep{N_EPISODES_TRAIN}_ep{EPOCHS}epochs")

    start_total = time.time()

    for epoch in range(start_epoch, EPOCHS):
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

        # Checkpoint toutes les 200 epochs
        if (epoch + 1) % 200 == 0:
            cp = save_checkpoint(model, optimizer, epoch, train_losses,
                                 test_losses, epoch_times, run_dir)
            # Copie numérotée pour garder l'historique
            import shutil
            shutil.copy(run_dir / "checkpoint.pt", run_dir / f"checkpoint_ep{epoch+1}.pt")
            print(f"  Checkpoint sauvegardé (epoch {epoch+1})")

    total_time = time.time() - start_total
    # Compter le temps des epochs précédentes (si reprise)
    total_time += sum(epoch_times[:start_epoch]) if start_epoch > 0 else 0
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
        action_buffer = []

        while step < MAX_STEPS:
            if ep < 3:
                frames.append(env.render())

            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                if norm:
                    agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")

                with torch.no_grad():
                    chunk_pred = model(img, agent_pos)

                if norm:
                    chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                else:
                    chunk_actions = chunk_pred.squeeze(0).numpy()
                action_buffer = list(chunk_actions)

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
    lr_str = f"_lr{LEARNING_RATE}" if LEARNING_RATE != 1e-3 else ""
    label = f"{N_LAYERS}layer_ep{N_EPISODES_TRAIN}_ep{EPOCHS}epochs{lr_str}"
    arch_name = f"CNN+Transformer {N_LAYERS}L {N_HEADS}H d={D_MODEL} chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    X_img, X_pos, Y, norm = load_data_chunked()

    # 2. Modèle
    model = TransformerChunkPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE,
    )
    info = model_info(model, name=arch_name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres : {info['total_params']:,}")

    # 3. FLOPs
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  FLOPs/appel  : {flops_inf:,}")
    print(f"  FLOPs/frame  : {flops_inf // CHUNK_SIZE:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y, resume_from=RESUME_FROM)

    # 5. Temps d'inférence
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  Inférence : {inf_time:.2f}ms/appel ({inf_time/CHUNK_SIZE:.2f}ms/frame)")

    # 6. Évaluation simulation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 7. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    model_config = {
        "class": "TransformerChunkPolicy",
        "cnn_channels": CNN_CHANNELS,
        "cnn_feature_dim": CNN_FEATURE_DIM,
        "image_size": IMAGE_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE,
        "pos_dim": 2,
        "action_dim": 2,
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
            "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "d_model": D_MODEL, "dim_feedforward": DIM_FEEDFORWARD,
            "chunk_size": CHUNK_SIZE,
            "description": f"Transformer decoder {N_LAYERS} couches + action chunking={CHUNK_SIZE}",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — Transformer {N_LAYERS} couches, chunk={CHUNK_SIZE}")
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
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers un checkpoint.pt pour reprendre")
    args = parser.parse_args()
    N_LAYERS = args.layers
    N_EPISODES_TRAIN = args.episodes
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    RESUME_FROM = args.resume
    main()
