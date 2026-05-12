"""
Expérience 30 — MDN (Mixture Density Network) sur image seule.

Variante de 29_image_only : on remplace la sortie regression + MSE par
une sortie "mélange de K gaussiennes" + NLL.

But : voir si modéliser explicitement la multimodalité (au lieu de
moyenner les actions valides) améliore le coverage, et si la loss
NLL corrèle mieux au coverage que la MSE.

Tout le reste identique au run 29 :
- Image seule (pas de agent_pos)
- Features ResNet18 pré-calculées
- Transformer 2 couches, mêmes hyperparams

Éval coverage à 4 checkpoints : epochs 25, 50, 75, 100.
"""

import sys
sys.path.insert(0, ".")

import math
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
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "30_mdn"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
FEATURE_DIM = 512

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
K_MIXTURES = 5  # Nombre de modes gaussiens
LOG_SIGMA_MIN = -5.0  # Clamp pour éviter mode collapse
LOG_SIGMA_MAX = 2.0

LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

N_EPISODES_TRAIN = 206
N_EVAL_EPISODES = 200
MAX_STEPS = 300
CHECKPOINT_EPOCHS = [25, 50, 75, 100]

CACHE_DIR = Path("data_cache")
# ============================================================


def extract_features():
    cache_file = CACHE_DIR / f"resnet18_features_ep{N_EPISODES_TRAIN}_{IMAGE_SIZE}px.pt"
    if cache_file.exists():
        print(f"  Features en cache : {cache_file}", flush=True)
        data = torch.load(cache_file, weights_only=True)
        return data["features"], data["positions"], data["actions"], data["norm"], data["episodes"]

    print("  Extraction des features ResNet18...", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    X_img, X_pos, Y, norm, meta = load_pusht_cached(
        n_episodes=N_EPISODES_TRAIN, image_size=IMAGE_SIZE, seq_len=1
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET, episodes=list(range(N_EPISODES_TRAIN)))
    all_eps = [dataset[i]["episode_index"].item() for i in range(len(dataset))]

    from torchvision.models import resnet18, ResNet18_Weights
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    features = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X_img), batch_size):
            batch = X_img[i:i + batch_size]
            feats = backbone(batch).flatten(1)
            features.append(feats)
    features = torch.cat(features)
    torch.save({
        "features": features, "positions": X_pos, "actions": Y,
        "norm": norm, "episodes": all_eps,
    }, cache_file)
    return features, X_pos, Y, norm, all_eps


class TransformerMDN(nn.Module):
    """Transformer image-only avec sortie MDN (mélange de K gaussiennes 2D diagonales).

    Sortie par timestep : K poids (logits), K×2 moyennes, K×2 log-σ.
    """

    def __init__(self, feature_dim, d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, k_mixtures, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.k = k_mixtures
        self.d_model = d_model

        self.obs_proj = nn.Linear(feature_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        # K weights + K * action_dim mu + K * action_dim log_sigma = K * (1 + 2*action_dim)
        self.mdn_head = nn.Linear(d_model, k_mixtures * (1 + 2 * action_dim))

    def forward(self, features):
        batch = features.shape[0]
        context = self.obs_proj(features).unsqueeze(1)
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)  # (batch, chunk, d_model)
        out = self.mdn_head(decoded)  # (batch, chunk, K * 5)
        out = out.view(batch, self.chunk_size, self.k, 1 + 2 * self.action_dim)
        pi_logits = out[..., 0]                                # (batch, chunk, K)
        mu = out[..., 1:1 + self.action_dim]                   # (batch, chunk, K, 2)
        log_sigma = out[..., 1 + self.action_dim:]             # (batch, chunk, K, 2)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return pi_logits, mu, log_sigma


def mdn_nll(pi_logits, mu, log_sigma, target):
    """NLL d'un mélange de K gaussiennes diagonales.

    pi_logits: (batch, chunk, K)
    mu: (batch, chunk, K, 2)
    log_sigma: (batch, chunk, K, 2)
    target: (batch, chunk, 2)
    """
    target_exp = target.unsqueeze(2)  # (batch, chunk, 1, 2)
    diff = target_exp - mu
    log_prob_per_dim = -0.5 * (diff * torch.exp(-log_sigma)) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    log_prob = log_prob_per_dim.sum(dim=-1)  # (batch, chunk, K)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)  # (batch, chunk)
    return -log_mixture.mean()


def mdn_sample(pi_logits, mu, log_sigma, mode='sample'):
    """Échantillonne dans le mélange. mode in {'sample', 'argmax', 'mean'}.

    Retourne (batch, chunk, 2).
    """
    batch, chunk, k = pi_logits.shape
    pi = F.softmax(pi_logits, dim=-1)
    if mode == 'argmax':
        idx = pi.argmax(dim=-1)  # (batch, chunk)
    else:
        idx = torch.multinomial(pi.reshape(-1, k), 1).reshape(batch, chunk)

    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.shape[-1])  # (batch, chunk, 1, 2)
    sel_mu = torch.gather(mu, dim=2, index=idx_exp).squeeze(2)
    sel_log_sigma = torch.gather(log_sigma, dim=2, index=idx_exp).squeeze(2)
    if mode == 'mean':
        return sel_mu
    return sel_mu + torch.randn_like(sel_mu) * torch.exp(sel_log_sigma)


def load_data_chunked(features, actions, episodes):
    valid_feats, valid_chunks = [], []
    i = 0
    while i + CHUNK_SIZE <= len(features):
        if episodes[i] == episodes[i + CHUNK_SIZE - 1]:
            valid_feats.append(features[i])
            valid_chunks.append(actions[i:i + CHUNK_SIZE])
            i += 1
        else:
            i += 1
    X_feat = torch.stack(valid_feats)
    Y = torch.stack(valid_chunks)
    print(f"  {len(X_feat)} samples (chunk={CHUNK_SIZE})", flush=True)
    return X_feat, Y


def train_model(model, X_feat, Y, norm):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n = len(X_feat)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_losses, test_losses = [], []
    epoch_times = []
    checkpoint_evals = []
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
            pi_logits, mu, log_sigma = model(X_feat[batch])
            loss = mdn_nll(pi_logits, mu, log_sigma, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        model.eval()
        with torch.no_grad():
            pi_logits, mu, log_sigma = model(X_feat[test_idx])
            test_loss = mdn_nll(pi_logits, mu, log_sigma, Y[test_idx]).item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train_nll: {avg_train:.4f} — test_nll: {test_loss:.4f} — {elapsed:.0f}s",
                  flush=True)

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            print(f"\n  [CHECKPOINT] Éval à epoch {epoch+1}...", flush=True)
            eval_res, _ = eval_in_simulation(model, norm, save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "test_loss": test_loss,
                "coverage": eval_res["avg_coverage"],
                "success_rate": eval_res["success_rate"],
            })
            print(f"  [CHECKPOINT epoch {epoch+1}] train={avg_train:.4f} "
                  f"test={test_loss:.4f} cov={eval_res['avg_coverage']:.1%} "
                  f"success={eval_res['success_rate']:.0%}", flush=True)

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s", flush=True)
    return train_losses, test_losses, epoch_times, total_time, len(train_idx), len(test_idx), checkpoint_evals


def eval_in_simulation(model, norm, save_videos=True):
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes)...", flush=True)

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
            if save_videos and ep < 3:
                frames.append(env.render())

            if len(action_buffer) == 0:
                img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img = F.interpolate(img, size=IMAGE_SIZE, mode="bilinear")
                with torch.no_grad():
                    features = backbone(img).flatten(1)
                    pi_logits, mu, log_sigma = model(features)
                    chunk_pred = mdn_sample(pi_logits, mu, log_sigma, mode='sample')
                chunk_actions = (chunk_pred.squeeze(0) * norm["y_std"] + norm["y_mean"]).numpy()
                action_buffer = list(chunk_actions)

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            obs, reward, _, _, info = env.step(action)
            ep_reward += reward
            max_coverage = max(max_coverage, info.get("coverage", 0))

        success = info.get("is_success", False)
        results.append({"reward": ep_reward, "success": success, "max_coverage": max_coverage})
        if (ep + 1) % 40 == 0:
            print(f"    Episode {ep+1:3d}/{N_EVAL_EPISODES} : cov={max_coverage:.1%}", flush=True)
        if frames:
            all_frames.append(frames)

    env.close()
    n_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_coverage = np.mean([r["max_coverage"] for r in results])
    print(f"    Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%}) "
          f"— Coverage moyen : {avg_coverage:.1%}", flush=True)
    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
    }, all_frames


def main():
    label = "resnet18_mdn"
    arch_name = f"ResNet18(gelé) → Transformer 2L → MDN(K={K_MIXTURES}) chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nFeatures ResNet18...", flush=True)
    features, positions, actions, norm, episodes = extract_features()
    X_feat, Y = load_data_chunked(features, actions, episodes)

    model = TransformerMDN(
        feature_dim=FEATURE_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE, k_mixtures=K_MIXTURES,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)

    print(f"\nEntraînement (MDN K={K_MIXTURES}, image seule)...", flush=True)
    train_losses, test_losses, epoch_times, train_time, n_train, n_test, checkpoint_evals = train_model(
        model, X_feat, Y, norm
    )

    # Inférence timing
    dummy_feat = torch.randn(1, FEATURE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            pi_logits, mu, log_sigma = model(dummy_feat)
            mdn_sample(pi_logits, mu, log_sigma, mode='sample')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            pi_logits, mu, log_sigma = model(dummy_feat)
            mdn_sample(pi_logits, mu, log_sigma, mode='sample')
    inf_time = (time.time() - t0) / 200 * 1000

    # Éval finale avec vidéos
    print("\nÉval finale avec vidéos...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, norm, save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "TransformerMDN",
        "feature_dim": FEATURE_DIM, "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE, "k_mixtures": K_MIXTURES,
        "log_sigma_min": LOG_SIGMA_MIN, "log_sigma_max": LOG_SIGMA_MAX,
        "note": "MDN sur image seule. Loss = NLL mélange K gaussiennes 2D diagonales.",
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    batches_per_epoch = n_train // BATCH_SIZE + 1
    flops_transformer = sum(2 * p.numel() for p in model.parameters())

    run_data = build_run_data(
        experiment=EXPERIMENT_NAME, architecture=arch_name, dataset=DATASET,
        n_episodes=N_EPISODES_TRAIN, n_params=info["total_params"], size_mb=info["size_mb"],
        seed=SEED, train_losses=train_losses, test_losses=test_losses,
        train_time_s=train_time, epoch_times=epoch_times,
        flops_total_train=flops_transformer * BATCH_SIZE * batches_per_epoch * EPOCHS,
        n_train_samples=n_train, n_test_samples=n_test,
        inference_time_ms=inf_time, flops_inference=flops_transformer,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "feature_extractor": "ResNet18 (gelé, pré-entraîné ImageNet)",
            "trainable_params": info["total_params"],
            "loss_type": "NLL_mixture_density",
            "k_mixtures": K_MIXTURES,
            "checkpoint_evals": checkpoint_evals,
            "description": "MDN sur image seule — modélise la multimodalité par mélange gaussien K=5.",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"RÉSUMÉ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Coverage final   : {eval_results['avg_coverage']:.1%}", flush=True)
    print(f"  Success rate     : {eval_results['success_rate']:.0%}", flush=True)
    print(f"  Params entraînés : {info['total_params']:,}", flush=True)
    print(f"  Temps train+eval : {train_time:.0f}s", flush=True)
    print(f"\n  Checkpoint evals (loss ↔ coverage) :", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} "
              f"test={cp['test_loss']:.4f} cov={cp['coverage']:.1%}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
