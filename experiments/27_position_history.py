"""
Expérience 27 — Historique des positions en entrée.

On donne au MLP les N dernières positions de l'agent en plus de l'image.
Ça ne coûte quasi rien en paramètres mais donne le contexte de mouvement.
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
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "27_pos_history"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
FEATURE_DIM = 512
HIDDEN_LAYERS = [256, 128]
CHUNK_SIZE = 20
POS_HISTORY = 5  # Nombre de positions passées

LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 1000
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 206
N_EVAL_EPISODES = 50
MAX_STEPS = 300

CACHE_DIR = Path("data_cache")
# ============================================================


class MLPChunkWithHistory(nn.Module):
    """MLP avec historique de positions.

    Entrée : features ResNet(512) + positions passées (2 × N) = 512 + 2*N
    Sortie : chunk de 20 actions (40 valeurs)
    """
    def __init__(self, feature_dim, hidden_layers, chunk_size, pos_history,
                 pos_dim=2, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.pos_history = pos_history

        input_dim = feature_dim + pos_dim * pos_history
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

    def forward(self, features, pos_history):
        # pos_history : (batch, pos_history * 2) — positions aplaties
        x = torch.cat([features, pos_history], dim=1)
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


def load_data_with_history(features, positions, actions, episodes):
    """Crée les samples avec historique de positions + chunk d'actions."""
    total_needed = POS_HISTORY + CHUNK_SIZE - 1

    valid_feats, valid_pos_hist, valid_chunks = [], [], []
    i = 0
    while i + total_needed <= len(features):
        # Vérifier que tout est dans le même épisode
        if episodes[i] == episodes[i + total_needed - 1]:
            # Frame courante = i + POS_HISTORY - 1
            current = i + POS_HISTORY - 1

            # Feature de la frame courante
            valid_feats.append(features[current])

            # Historique des positions : [pos_t-N+1, ..., pos_t]
            pos_hist = positions[i:i + POS_HISTORY].reshape(-1)  # (POS_HISTORY * 2,)
            valid_pos_hist.append(pos_hist)

            # Chunk d'actions à partir de la frame courante
            valid_chunks.append(actions[current:current + CHUNK_SIZE])

            i += 1
        else:
            i += 1

    X_feat = torch.stack(valid_feats)
    X_pos = torch.stack(valid_pos_hist)
    Y = torch.stack(valid_chunks)

    print(f"  {len(X_feat)} samples (history={POS_HISTORY}, chunk={CHUNK_SIZE})")
    print(f"  Entrée : features(512) + positions({POS_HISTORY}×2={POS_HISTORY*2}) = {512 + POS_HISTORY*2}")
    return X_feat, X_pos, Y


def train_model(model, X_feat, X_pos, Y):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X_feat)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    epoch_times = []
    start_total = time.time()

    for epoch in range(EPOCHS):
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

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:4d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s")

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s")
    return train_losses, test_losses, epoch_times, total_time, len(train_idx), len(test_idx)


def eval_in_simulation(model, norm):
    import gymnasium as gym
    import gym_pusht  # noqa
    from collections import deque

    print(f"\n  Évaluation ({N_EVAL_EPISODES} épisodes, history={POS_HISTORY})...")

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

        # Buffer de positions passées
        pos_buffer = deque(maxlen=POS_HISTORY)

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            # Ajouter la position actuelle au buffer
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            agent_pos_norm = (agent_pos - norm["p_mean"]) / norm["p_std"]
            pos_buffer.append(agent_pos_norm)

            if len(action_buffer) == 0:
                # Remplir le buffer si pas assez de positions
                while len(pos_buffer) < POS_HISTORY:
                    pos_buffer.appendleft(pos_buffer[0])

                # Image → ResNet
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")
                with torch.no_grad():
                    features = backbone(img).flatten(1)  # (1, 512)

                # Historique des positions aplati
                pos_hist = torch.cat(list(pos_buffer)).unsqueeze(0)  # (1, POS_HISTORY*2)

                with torch.no_grad():
                    chunk_pred = model(features, pos_hist)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, _, _, info = env.step(action)
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


def main():
    label = f"hist{POS_HISTORY}"
    arch_name = f"ResNet18 → MLP {HIDDEN_LAYERS} hist={POS_HISTORY} chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    print("\nFeatures ResNet18...")
    features, positions, actions, norm, episodes = load_features()
    X_feat, X_pos, Y = load_data_with_history(features, positions, actions, episodes)

    # 2. Modèle
    model = MLPChunkWithHistory(
        feature_dim=FEATURE_DIM, hidden_layers=HIDDEN_LAYERS,
        chunk_size=CHUNK_SIZE, pos_history=POS_HISTORY,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres : {info['total_params']:,}")

    # 3. Entraîner
    print(f"\nEntraînement ({EPOCHS} epochs)...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_feat, X_pos, Y)

    # 4. Évaluation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 5. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "MLPChunkWithHistory",
        "feature_dim": FEATURE_DIM, "hidden_layers": HIDDEN_LAYERS,
        "chunk_size": CHUNK_SIZE, "pos_history": POS_HISTORY,
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
        train_time_s=train_time,
        epoch_times=epoch_times,
        flops_total_train=0,
        n_train_samples=n_train,
        n_test_samples=n_test,
        inference_time_ms=0,
        flops_inference=0,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "pos_history": POS_HISTORY,
            "description": f"MLP avec {POS_HISTORY} positions passées en entrée",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — history={POS_HISTORY}")
    print(f"{'='*60}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Params        : {info['total_params']:,}")
    print(f"  Temps train   : {train_time:.0f}s")
    print(f"  Dossier       : {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    POS_HISTORY = args.history
    EPOCHS = args.epochs
    main()
