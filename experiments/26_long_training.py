"""
Expérience 26 — Entraînement long (10 000 epochs) avec régularisation.

ResNet18 features + MLP avec dropout + weight decay.
Évaluation du coverage toutes les 100 epochs pour voir l'évolution.
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
import matplotlib.pyplot as plt

from pathlib import Path
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "26_long_training"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
FEATURE_DIM = 512
HIDDEN_LAYERS = [256, 128]
CHUNK_SIZE = 20
DROPOUT = 0.1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 100
GRAD_CLIP = 1.0
EPOCHS = 10000
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 206
N_EVAL_EPISODES = 50
MAX_STEPS = 300
EVAL_EVERY = 500  # Évaluer le coverage toutes les N epochs

CACHE_DIR = Path("data_cache")
# ============================================================


class MLPChunkWithDropout(nn.Module):
    """MLP avec dropout pour la régularisation."""

    def __init__(self, feature_dim, hidden_layers, chunk_size, dropout=0.1,
                 pos_dim=2, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        layers = []
        prev = feature_dim + pos_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

    def forward(self, features, agent_pos):
        x = torch.cat([features, agent_pos], dim=1)
        out = self.net(x)
        return out.reshape(-1, self.chunk_size, self.action_dim)


def load_features():
    cache_file = CACHE_DIR / f"resnet18_features_ep{N_EPISODES_TRAIN}_{IMAGE_SIZE}px.pt"
    if not cache_file.exists():
        print("  Extraction features ResNet18...")
        from src.data_cache import load_pusht_cached
        X_img, X_pos, Y, norm, meta = load_pusht_cached(
            n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
        )
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.eval()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        features = []
        with torch.no_grad():
            for i in range(0, len(X_img), 128):
                features.append(backbone(X_img[i:i+128]).flatten(1))
        features = torch.cat(features)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(DATASET, episodes=list(range(N_EPISODES_TRAIN)))
        all_eps = [dataset[i]["episode_index"].item() for i in range(len(dataset))]
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save({"features": features, "positions": X_pos, "actions": Y,
                     "norm": norm, "episodes": all_eps}, cache_file)
    print(f"  Cache : {cache_file}")
    data = torch.load(cache_file, weights_only=True)
    return data["features"], data["positions"], data["actions"], data["norm"], data["episodes"]


def load_data_chunked(features, positions, actions, episodes):
    valid_feats, valid_pos, valid_chunks = [], [], []
    i = 0
    while i + CHUNK_SIZE <= len(features):
        if episodes[i] == episodes[i + CHUNK_SIZE - 1]:
            valid_feats.append(features[i])
            valid_pos.append(positions[i])
            valid_chunks.append(actions[i:i + CHUNK_SIZE])
            i += 1
        else:
            i += 1
    return torch.stack(valid_feats), torch.stack(valid_pos), torch.stack(valid_chunks)


def eval_coverage(model, norm):
    """Évaluation rapide du coverage (sans vidéos)."""
    import gymnasium as gym
    import gym_pusht  # noqa

    from torchvision.models import resnet18, ResNet18_Weights
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    coverages = []
    model.eval()

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        max_coverage = 0
        action_buffer = []

        for step in range(MAX_STEPS):
            if len(action_buffer) == 0:
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
                agent_pos = (agent_pos - norm["p_mean"]) / norm["p_std"]
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")
                with torch.no_grad():
                    features = backbone(img).flatten(1)
                    chunk_pred = model(features, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, _, _, info = env.step(action)
            max_coverage = max(max_coverage, info.get("coverage", 0))

        coverages.append(max_coverage)

    env.close()
    return np.mean(coverages)


def eval_with_videos(model, norm):
    """Évaluation complète avec vidéos (pour la fin)."""
    import gymnasium as gym
    import gym_pusht  # noqa

    from torchvision.models import resnet18, ResNet18_Weights
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

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
                    features = backbone(img).flatten(1)
                    chunk_pred = model(features, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, _, _, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        if ep < 3:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_coverage = np.mean([r["max_coverage"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
    }, all_frames


def main():
    label = f"dropout{DROPOUT}_wd{WEIGHT_DECAY}"
    arch_name = f"ResNet18 → MLP {HIDDEN_LAYERS} drop={DROPOUT} chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"{'='*60}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Dropout    : {DROPOUT}")
    print(f"  Weight dec : {WEIGHT_DECAY}")
    print(f"  Warmup     : {WARMUP_EPOCHS} epochs")
    print(f"  Eval every : {EVAL_EVERY} epochs")

    torch.manual_seed(SEED)

    # 1. Charger
    print("\nFeatures ResNet18...")
    features, positions, actions, norm, episodes = load_features()
    X_feat, X_pos, Y = load_data_chunked(features, positions, actions, episodes)
    print(f"  {len(X_feat)} samples")

    # 2. Modèle
    model = MLPChunkWithDropout(
        feature_dim=FEATURE_DIM, hidden_layers=HIDDEN_LAYERS,
        chunk_size=CHUNK_SIZE, dropout=DROPOUT,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres : {info['total_params']:,}")

    # 3. Entraîner avec évals intermédiaires
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    n = len(X_feat)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    epoch_times = []
    coverage_history = []  # (epoch, coverage)
    start_total = time.time()

    print(f"\nEntraînement ({EPOCHS} epochs)...")

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
            pred = model(X_feat[batch], X_pos[batch])
            loss = criterion(pred, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            test_pred = model(X_feat[test_idx], X_pos[test_idx])
            test_loss = criterion(test_pred, Y[test_idx]).item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:5d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s")

        # Éval coverage toutes les EVAL_EVERY epochs
        if (epoch + 1) % EVAL_EVERY == 0:
            coverage = eval_coverage(model, norm)
            coverage_history.append((epoch + 1, coverage))
            elapsed = time.time() - start_total
            print(f"  >>> Epoch {epoch+1} : coverage = {coverage:.1%} ({elapsed:.0f}s)")

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s")

    # 4. Éval finale avec vidéos
    print("\nÉvaluation finale...")
    eval_results, all_frames = eval_with_videos(model, norm)
    print(f"  Coverage : {eval_results['avg_coverage']:.1%}")
    print(f"  Success  : {eval_results['success_rate']:.0%}")

    # 5. Graphes
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    # Graphe loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train", linewidth=1, alpha=0.7)
    ax1.plot(test_losses, label="Test", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graphe coverage
    epochs_cov = [c[0] for c in coverage_history]
    coverages_cov = [c[1] * 100 for c in coverage_history]
    ax2.plot(epochs_cov, coverages_cov, "o-", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Coverage max moyen (%)")
    ax2.set_title("Coverage vs Epoch")
    ax2.grid(True, alpha=0.3)
    if coverages_cov:
        best_idx = coverages_cov.index(max(coverages_cov))
        ax2.axvline(x=epochs_cov[best_idx], color="red", linestyle="--", alpha=0.5,
                     label=f"Best: {coverages_cov[best_idx]:.1f}% @ ep{epochs_cov[best_idx]}")
        ax2.legend()

    plt.suptitle(f"{EXPERIMENT_NAME} — {arch_name}")
    plt.tight_layout()
    plt.savefig(str(run_dir / "losses.png"), dpi=150)
    plt.close()

    # 6. Sauvegarder
    model_config = {
        "class": "MLPChunkWithDropout",
        "feature_dim": FEATURE_DIM, "hidden_layers": HIDDEN_LAYERS,
        "chunk_size": CHUNK_SIZE, "dropout": DROPOUT,
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

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
        train_time_s=total_time,
        epoch_times=epoch_times,
        flops_total_train=0,
        n_train_samples=len(train_idx),
        n_test_samples=len(test_idx),
        inference_time_ms=0,
        flops_inference=0,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
            "coverage_history": coverage_history,
            "best_coverage": max(coverages_cov) / 100 if coverages_cov else 0,
            "best_epoch": epochs_cov[best_idx] if coverages_cov else 0,
            "description": f"10k epochs, dropout={DROPOUT}, wd={WEIGHT_DECAY}",
        },
    )
    save_run(run_data)
    generate_run_info(run_data, model, run_dir)

    # Résumé
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  Coverage final : {eval_results['avg_coverage']:.1%}")
    if coverages_cov:
        print(f"  Meilleur coverage : {max(coverages_cov):.1f}% à l'epoch {epochs_cov[best_idx]}")
    print(f"  Coverage history :")
    for ep, cov in coverage_history:
        print(f"    Epoch {ep:5d} : {cov:.1%}")
    print(f"  Temps total : {total_time:.0f}s")
    print(f"  Dossier     : {run_dir}")


if __name__ == "__main__":
    main()
