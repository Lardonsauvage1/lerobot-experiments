"""
Expérience 31 — Discrétisation + Cross-Entropy (BeT-simple) sur image seule.

Variante de 29_image_only : on remplace la régression MSE par
une classification sur N bins d'actions (issus d'un k-means sur le
train set) + un résidu continu par bin (style BeT).

But : voir si transformer la régression en classification + résidu
améliore le coverage en évitant le mode collapse, et si la loss
CE+résidu corrèle mieux au coverage que la MSE.

Tout le reste identique au run 29.

Améliorations vs 30 :
- bin_centers stockés comme buffer dans le modèle (state_dict propre)
- K-means sur train set uniquement (pas de fuite test)
- Éval checkpoints en mode argmax (déterministe → corrélation propre)
  Éval finale en mode sample (capture la multimodalité)
- Sauvegarde du best checkpoint en plus du final

Éval coverage à 4 checkpoints : epochs 25, 50, 75, 100.
"""

import sys
sys.path.insert(0, ".")

import copy
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
EXPERIMENT_NAME = "31_discrete_ce"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
FEATURE_DIM = 512

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
N_BINS = 64
RESIDUAL_WEIGHT = 1.0

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


def compute_bins(train_actions, n_bins):
    """K-means sur les actions du train set uniquement.

    train_actions: (N_train, 2) — actions plates du train set (pas de chunks).
    """
    cache = CACHE_DIR / f"action_bins_n{n_bins}_seed{SEED}_train.pt"
    if cache.exists():
        print(f"  Bins en cache : {cache}", flush=True)
        return torch.load(cache, weights_only=True)

    from sklearn.cluster import KMeans
    actions_np = train_actions.numpy()
    print(f"  K-means N={n_bins} sur {len(actions_np)} actions (train only)...", flush=True)
    kmeans = KMeans(n_clusters=n_bins, random_state=SEED, n_init=10).fit(actions_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    torch.save(centers, cache)
    print(f"  Centres bins shape : {centers.shape}", flush=True)
    return centers


def assign_bins(actions, centers):
    """Pour chaque action, trouve l'index du bin le plus proche."""
    flat = actions.reshape(-1, 2)
    dists = torch.cdist(flat, centers)
    idx = dists.argmin(dim=-1)
    return idx.reshape(actions.shape[:-1])


class TransformerDiscrete(nn.Module):
    """Transformer image-only avec sortie discrète + résidu (style BeT)."""

    def __init__(self, feature_dim, d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, n_bins, bin_centers, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = n_bins

        self.register_buffer("bin_centers", bin_centers)

        self.obs_proj = nn.Linear(feature_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.bin_head = nn.Linear(d_model, n_bins)
        self.residual_head = nn.Linear(d_model, n_bins * action_dim)

    def forward(self, features):
        batch = features.shape[0]
        context = self.obs_proj(features).unsqueeze(1)
        queries = (self.query_tokens + self.pos_encoding).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, context)
        bin_logits = self.bin_head(decoded)
        residuals = self.residual_head(decoded).view(
            batch, self.chunk_size, self.n_bins, self.action_dim
        )
        return bin_logits, residuals

    def compute_loss(self, bin_logits, residuals, target):
        batch, chunk, n_bins = bin_logits.shape
        target_bins = assign_bins(target, self.bin_centers)
        ce_loss = F.cross_entropy(
            bin_logits.reshape(-1, n_bins),
            target_bins.reshape(-1),
        )
        target_residuals = target - self.bin_centers[target_bins]
        idx = target_bins.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        sel_residual = torch.gather(residuals, dim=2, index=idx).squeeze(2)
        mse_loss = F.mse_loss(sel_residual, target_residuals)
        return ce_loss + RESIDUAL_WEIGHT * mse_loss, ce_loss.item(), mse_loss.item()

    def predict(self, bin_logits, residuals, mode='sample'):
        """mode in {'sample', 'argmax'}."""
        batch, chunk, n_bins = bin_logits.shape
        if mode == 'argmax':
            bin_ids = bin_logits.argmax(dim=-1)
        else:
            probs = F.softmax(bin_logits, dim=-1)
            bin_ids = torch.multinomial(probs.reshape(-1, n_bins), 1).reshape(batch, chunk)
        centers = self.bin_centers[bin_ids]
        idx = bin_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        sel_residual = torch.gather(residuals, dim=2, index=idx).squeeze(2)
        return centers + sel_residual


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


def train_model(model, X_feat, Y, train_idx, test_idx, norm):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, test_losses = [], []
    ce_train, mse_train = [], []
    epoch_times = []
    checkpoint_evals = []
    best_coverage = -1.0
    best_state_dict = None
    best_epoch = None
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
        epoch_loss, epoch_ce, epoch_mse, n_batches = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            bin_logits, residuals = model(X_feat[batch])
            loss, ce_v, mse_v = model.compute_loss(bin_logits, residuals, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ce += ce_v
            epoch_mse += mse_v
            n_batches += 1

        avg_train = epoch_loss / n_batches
        avg_ce = epoch_ce / n_batches
        avg_mse = epoch_mse / n_batches

        model.eval()
        with torch.no_grad():
            bin_logits, residuals = model(X_feat[test_idx])
            test_loss, _, _ = model.compute_loss(bin_logits, residuals, Y[test_idx])
            test_loss = test_loss.item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        ce_train.append(avg_ce)
        mse_train.append(avg_mse)
        epoch_times.append(ep_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train: {avg_train:.4f} (ce={avg_ce:.4f} mse={avg_mse:.4f}) — "
                  f"test: {test_loss:.4f} — {elapsed:.0f}s", flush=True)

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            print(f"\n  [CHECKPOINT] Éval à epoch {epoch+1} (mode=argmax)...", flush=True)
            eval_res, _ = eval_in_simulation(model, norm, eval_mode='argmax', save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "test_loss": test_loss,
                "ce_loss": avg_ce,
                "mse_loss": avg_mse,
                "coverage": eval_res["avg_coverage"],
                "success_rate": eval_res["success_rate"],
            })
            print(f"  [CHECKPOINT epoch {epoch+1}] train={avg_train:.4f} "
                  f"test={test_loss:.4f} cov={eval_res['avg_coverage']:.1%} "
                  f"success={eval_res['success_rate']:.0%}", flush=True)

            if eval_res["avg_coverage"] > best_coverage:
                best_coverage = eval_res["avg_coverage"]
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"  [NEW BEST] epoch {epoch+1} : coverage {best_coverage:.1%}", flush=True)

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s", flush=True)
    return (train_losses, test_losses, ce_train, mse_train, epoch_times,
            total_time, len(train_idx), len(test_idx), checkpoint_evals,
            best_state_dict, best_coverage, best_epoch)


def eval_in_simulation(model, norm, eval_mode='sample', save_videos=True):
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes, mode={eval_mode})...", flush=True)

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
                    bin_logits, residuals = model(features)
                    chunk_pred = model.predict(bin_logits, residuals, mode=eval_mode)
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
    label = "resnet18_discrete_ce"
    arch_name = f"ResNet18(gelé) → Transformer 2L → Discret({N_BINS} bins + résidu) chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nFeatures ResNet18...", flush=True)
    features, positions, actions, norm, episodes = extract_features()
    X_feat, Y = load_data_chunked(features, actions, episodes)

    # Split d'abord pour pouvoir calculer les bins sur train uniquement
    n = len(X_feat)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    # K-means sur les actions du TRAIN uniquement (pas de fuite test)
    print("\nCalcul des bins (k-means sur train)...", flush=True)
    # Pour éviter le biais d'overlapping chunks, prendre seulement la 1re action de chaque chunk
    train_actions_flat = Y[train_idx][:, 0, :]  # (N_train, 2)
    bin_centers = compute_bins(train_actions_flat, N_BINS)

    model = TransformerDiscrete(
        feature_dim=FEATURE_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE, n_bins=N_BINS, bin_centers=bin_centers,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)

    print(f"\nEntraînement (Discret N={N_BINS} + résidu, image seule)...", flush=True)
    (train_losses, test_losses, ce_train, mse_train, epoch_times,
     train_time, n_train, n_test, checkpoint_evals,
     best_state_dict, best_coverage, best_epoch) = train_model(
        model, X_feat, Y, train_idx, test_idx, norm
    )

    dummy_feat = torch.randn(1, FEATURE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            bin_logits, residuals = model(dummy_feat)
            model.predict(bin_logits, residuals, mode='sample')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            bin_logits, residuals = model(dummy_feat)
            model.predict(bin_logits, residuals, mode='sample')
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale (mode=sample, capture la multimodalité)...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, norm, eval_mode='sample', save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "TransformerDiscrete",
        "feature_dim": FEATURE_DIM, "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE, "n_bins": N_BINS,
        "note": "Discret + résidu (BeT-simple) sur image seule. K-means sur train. bin_centers en buffer.",
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    # Sauvegarde du best checkpoint
    if best_state_dict is not None:
        best_path = run_dir / f"model_best_epoch{best_epoch}.pt"
        torch.save({
            "state_dict": best_state_dict,
            "epoch": best_epoch,
            "coverage": best_coverage,
            "config": model_config,
        }, best_path)
        print(f"  Best model sauvegardé : {best_path} (epoch {best_epoch}, cov {best_coverage:.1%})", flush=True)

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
            "loss_type": "cross_entropy_plus_mse_residual",
            "n_bins": N_BINS,
            "residual_weight": RESIDUAL_WEIGHT,
            "ce_train_per_epoch": ce_train,
            "mse_train_per_epoch": mse_train,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch,
            "best_checkpoint_coverage": best_coverage,
            "kmeans_on_train_only": True,
            "checkpoint_eval_mode": "argmax",
            "final_eval_mode": "sample",
            "description": "BeT-simple : k-means N=64 train-only + CE sur bin + MSE sur résidu par bin. bin_centers en buffer. Best model sauvegardé.",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"RÉSUMÉ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Coverage final (sample)  : {eval_results['avg_coverage']:.1%}", flush=True)
    print(f"  Best checkpoint (argmax) : {best_coverage:.1%} (epoch {best_epoch})", flush=True)
    print(f"  Success rate             : {eval_results['success_rate']:.0%}", flush=True)
    print(f"  Params entraînés         : {info['total_params']:,}", flush=True)
    print(f"  Temps train+eval         : {train_time:.0f}s", flush=True)
    print(f"\n  Checkpoint evals (argmax, déterministe) :", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} "
              f"(ce={cp['ce_loss']:.4f} mse={cp['mse_loss']:.4f}) "
              f"test={cp['test_loss']:.4f} cov={cp['coverage']:.1%}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
