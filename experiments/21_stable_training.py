"""
Expérience 21 — Stabiliser l'entraînement avec lr warmup + gradient clipping.

On compare avec l'exp 20 (même modèle, mêmes 3 seeds, mais sans stabilisation).
100 epochs pour aller vite.

Ajouts :
  - LR warmup : lr monte de 0 à 1e-3 sur les 10 premières epochs
  - Gradient clipping : norme max = 1.0
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

from pathlib import Path
from src.data_cache import load_pusht_cached
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, count_flops, measure_inference_time
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "21_stable_training"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
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
        # LR warmup
        if epoch < WARMUP_EPOCHS:
            lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        else:
            lr = LEARNING_RATE
        for pg in optimizer.param_groups:
            pg["lr"] = lr

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
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — lr: {lr:.6f} — {elapsed:.0f}s")

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s")
    return train_losses, test_losses, epoch_times, total_time, len(train_idx), len(test_idx)


def eval_in_simulation(model, norm):
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Évaluation ({N_EVAL_EPISODES} épisodes)...")
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
        if (ep + 1) % 10 == 0:
            status = "✓" if success else "✗"
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


def run_with_seed(seed):
    global SEED
    SEED = seed
    label = f"seed{seed}"
    arch_name = f"Transformer 2L chunk={CHUNK_SIZE} warmup+clip"

    print(f"\n{'='*60}")
    print(f"SEED = {seed} (warmup={WARMUP_EPOCHS}ep, grad_clip={GRAD_CLIP})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    X_img, X_pos, Y, norm = load_data_chunked()

    model = TransformerChunkPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE,
    )
    info = model_info(model, name=f"{arch_name} seed={seed}")

    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })

    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y)

    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })

    eval_results, all_frames = eval_in_simulation(model, norm)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "TransformerChunkPolicy",
        "cnn_channels": CNN_CHANNELS, "cnn_feature_dim": CNN_FEATURE_DIM,
        "image_size": IMAGE_SIZE, "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE,
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
        seed=seed,
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
            "seed": seed, "warmup_epochs": WARMUP_EPOCHS, "grad_clip": GRAD_CLIP,
            "description": f"Transformer 2L chunk=20, warmup+clip, seed={seed}",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n  RÉSULTAT seed={seed} : coverage={eval_results['avg_coverage']:.1%}, "
          f"success={eval_results['success_rate']:.0%}")
    return eval_results


def main():
    seeds = [42, 123, 777]
    all_results = []

    for seed in seeds:
        result = run_with_seed(seed)
        all_results.append({"seed": seed, **result})

    print(f"\n{'='*60}")
    print(f"COMPARAISON — Avec warmup+clip (100 epochs)")
    print(f"{'='*60}")
    print(f"  {'Seed':>6}  {'Coverage':>10}  {'Success':>10}")
    for r in all_results:
        print(f"  {r['seed']:>6}  {r['avg_coverage']:>9.1%}  {r['success_rate']:>9.0%}")

    coverages = [r["avg_coverage"] for r in all_results]
    print(f"\n  Moyenne    : {np.mean(coverages):.1%}")
    print(f"  Écart-type : {np.std(coverages):.1%}")
    print(f"  Min/Max    : {min(coverages):.1%} / {max(coverages):.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    EPOCHS = args.epochs
    main()
