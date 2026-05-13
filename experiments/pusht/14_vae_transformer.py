"""
Expérience 14 — Transformer + VAE pour l'action chunking.

Le VAE résout le problème de multimodalité : quand il y a plusieurs
façons de faire la tâche, au lieu de prédire la moyenne (qui ne marche pas),
le VAE choisit UNE stratégie cohérente.

Entraînement :
  Actions futures (vraies) → VAE Encoder → z (résumé de la stratégie)
  Image + position + z → Transformer Decoder → actions prédites
  Loss = MSE(actions) + KL_weight * KL(z, N(0,1))

Inférence :
  z ~ N(0,1) (tiré aléatoirement)
  Image + position + z → Transformer Decoder → actions cohérentes
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
# PARAMÈTRES
# ============================================================

EXPERIMENT_NAME = "14_vae_transformer"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
CNN_CHANNELS = [16, 32]
CNN_FEATURE_DIM = 64

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
Z_DIM = 16
KL_WEIGHT = 0.01

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 206
N_EVAL_EPISODES = 10
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


class VAEEncoder(nn.Module):
    """Encode une séquence d'actions en un vecteur latent z.

    Pendant l'entraînement, le VAE Encoder voit les VRAIES actions futures
    et les résume en z = "cette démonstration pousse par la gauche".

    Produit mu et logvar pour l'échantillonnage avec le reparameterization trick.
    """

    def __init__(self, action_dim, chunk_size, z_dim):
        super().__init__()
        input_dim = action_dim * chunk_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

    def forward(self, actions_flat):
        # actions_flat : (batch, chunk_size * action_dim)
        h = self.net(actions_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAETransformerPolicy(nn.Module):
    """CNN + VAE + Transformer Decoder pour l'action chunking.

    Entraînement :
        Image → CNN → features
        Actions futures (vraies) → VAE Encoder → z (mu, logvar)
        z échantillonné + features + position → Transformer → actions prédites

    Inférence :
        Image → CNN → features
        z ~ N(0, 1) (aléatoire)
        z + features + position → Transformer → actions
    """

    def __init__(self, cnn_channels, cnn_feature_dim, image_size,
                 d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, z_dim, pos_dim=2, action_dim=2):
        super().__init__()
        self.cnn = SimpleCNN(feature_dim=cnn_feature_dim, image_size=image_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.d_model = d_model

        # VAE Encoder
        self.vae_encoder = VAEEncoder(action_dim, chunk_size, z_dim)

        # Projection observation + z → d_model
        self.obs_proj = nn.Linear(cnn_feature_dim + pos_dim + z_dim, d_model)

        # Query tokens apprenables
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Projection sortie
        self.action_head = nn.Linear(d_model, action_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick : z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, image, agent_pos, target_actions=None):
        """
        En entraînement (target_actions fourni) :
            Utilise le VAE encoder pour obtenir z à partir des vraies actions.
            Retourne (actions_prédites, mu, logvar)

        En inférence (target_actions=None) :
            Échantillonne z ~ N(0, 1)
            Retourne actions_prédites
        """
        batch = image.shape[0]

        # CNN
        cnn_feats = self.cnn(image)

        # VAE
        if target_actions is not None:
            # Entraînement : encoder les vraies actions
            actions_flat = target_actions.reshape(batch, -1)
            mu, logvar = self.vae_encoder(actions_flat)
            z = self.reparameterize(mu, logvar)
        else:
            # Inférence : z aléatoire
            z = torch.randn(batch, self.z_dim, device=image.device)
            mu, logvar = None, None

        # Concaténer observation + z
        obs = torch.cat([cnn_feats, agent_pos, z], dim=1)
        context = self.obs_proj(obs).unsqueeze(1)  # (batch, 1, d_model)

        # Transformer Decoder
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)
        actions = self.action_head(decoded)  # (batch, chunk_size, action_dim)

        if target_actions is not None:
            return actions, mu, logvar
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


def train_model(model, X_img, X_pos, Y):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            actions_pred, mu, logvar = model(X_img[batch], X_pos[batch],
                                              target_actions=Y[batch])

            # Loss = MSE + KL divergence (avec annealing)
            mse_loss = F.mse_loss(actions_pred, Y[batch])
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # KL annealing : kl_weight augmente linéairement de 0 à KL_WEIGHT
            kl_w = KL_WEIGHT * min(1.0, epoch / (EPOCHS * 0.5))  # atteint KL_WEIGHT à mi-entraînement
            loss = mse_loss + kl_w * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            # Test : on utilise aussi le VAE encoder (pas l'inférence aléatoire)
            test_pred, test_mu, test_logvar = model(X_img[test_idx], X_pos[test_idx],
                                                      target_actions=Y[test_idx])
            mse_test = F.mse_loss(test_pred, Y[test_idx]).item()
            kl_test = -0.5 * torch.mean(1 + test_logvar - test_mu.pow(2) - test_logvar.exp()).item()
            test_loss = mse_test + KL_WEIGHT * kl_test

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            current_kl_w = KL_WEIGHT * min(1.0, epoch / (EPOCHS * 0.5))
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.6f} — test: {test_loss:.6f} "
                  f"(mse: {mse_test:.6f}, kl: {kl_test:.4f}, kl_w: {current_kl_w:.4f}) — {elapsed:.0f}s")

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
                    # Inférence : z aléatoire (pas de target_actions)
                    chunk_pred = model(img, agent_pos)
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
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
    label = f"vae_z{Z_DIM}_kl{KL_WEIGHT}_annealing"
    arch_name = f"CNN+VAE(z={Z_DIM})+Transformer {N_LAYERS}L chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME} — {label}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # 1. Charger
    X_img, X_pos, Y, norm = load_data_chunked()

    # 2. Modèle
    model = VAETransformerPolicy(
        cnn_channels=CNN_CHANNELS, cnn_feature_dim=CNN_FEATURE_DIM,
        image_size=IMAGE_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE, z_dim=Z_DIM,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres : {info['total_params']:,}")
    print(f"  z_dim      : {Z_DIM}")
    print(f"  KL weight  : {KL_WEIGHT}")

    # 3. FLOPs
    flops_inf = count_flops(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })
    print(f"  FLOPs/appel  : {flops_inf:,}")

    # 4. Entraîner
    print("\nEntraînement...")
    train_losses, test_losses, epoch_times, train_time, n_train, n_test = train_model(model, X_img, X_pos, Y)

    # 5. Inférence
    inf_time = measure_inference_time(model, {
        "image": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "agent_pos": torch.randn(1, 2),
    })

    # 6. Évaluation
    eval_results, all_frames = eval_in_simulation(model, norm)

    # 7. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, label)

    model_config = {
        "class": "VAETransformerPolicy",
        "cnn_channels": CNN_CHANNELS, "cnn_feature_dim": CNN_FEATURE_DIM,
        "image_size": IMAGE_SIZE, "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE, "z_dim": Z_DIM,
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
            "z_dim": Z_DIM, "kl_weight": KL_WEIGHT,
            "description": f"Transformer + VAE (z={Z_DIM}) + action chunking={CHUNK_SIZE}",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — VAE + Transformer, chunk={CHUNK_SIZE}")
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
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=206)
    parser.add_argument("--z-dim", type=int, default=16)
    args = parser.parse_args()
    KL_WEIGHT = args.kl_weight
    EPOCHS = args.epochs
    N_EPISODES_TRAIN = args.episodes
    Z_DIM = args.z_dim
    main()
