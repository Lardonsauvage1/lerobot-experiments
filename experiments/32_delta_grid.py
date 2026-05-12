"""
Expérience 32 — Bins en grille uniforme sur les deltas humains.

Variante de 31_discrete_ce :
- Au lieu de cluster des actions absolues, on cluster des **deltas** action→action.
- Au lieu d'un k-means, on utilise une **grille uniforme 8×8** (= 64 bins),
  hand-designed à partir du prior "symétrique, borné".
- Le modèle prédit 20 deltas, on reconstruit cumulativement à l'inférence.

But : briser le saccadé du run 31 (téléportations entre bins absolus
éloignés) en passant à un contrôle en vitesse plutôt qu'en position.

Tout le reste identique au run 31 :
- Image seule (pas de agent_pos en input)
- Features ResNet18 pré-calculées, Transformer 2L
- 100 epochs, checkpoints 25/50/75/100 (argmax)
- Final eval en sample + vidéos
- Best model sauvegardé

Note "delta_0" : on cluster uniquement les vrais deltas humains
(a_i - a_{i-1}). On skip les chunks i=0 d'un épisode au training
(pas de a_{i-1}). À l'inférence, bootstrap au step 0 d'épisode :
on commande l'agent à sa propre position (pas de mouvement), puis
on enchaîne avec les deltas prédits.
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
EXPERIMENT_NAME = "32_delta_grid"
DATASET = "lerobot/pusht"

IMAGE_SIZE = 64
FEATURE_DIM = 512

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 256

CHUNK_SIZE = 20
N_BINS_PER_AXIS = 8        # → 64 bins total
N_BINS = N_BINS_PER_AXIS ** 2
GRID_PERCENTILE = 99.5     # bornes de la grille = ±P99.5 des deltas train
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


def load_data_chunked_delta(features, actions, episodes):
    """Construit les chunks de 20 actions, avec deltas action→action.

    Pour chaque chunk démarrant à l'index i :
      - feature   = features[i]
      - deltas[j] = actions[i+j] - actions[i+j-1]  (en espace action normalisé)

    On skip les chunks où i=0 dans l'épisode (pas de a_{i-1}).
    """
    episodes_arr = np.array(episodes)
    valid_feats, valid_deltas = [], []
    n = len(features)
    i = 0
    while i + CHUNK_SIZE <= n:
        # Skip si on est au tout début d'un épisode (pas de a_{i-1})
        if i == 0 or episodes_arr[i] != episodes_arr[i - 1]:
            i += 1
            continue
        # Skip si le chunk traverse une frontière d'épisode
        if episodes_arr[i] != episodes_arr[i + CHUNK_SIZE - 1]:
            i += 1
            continue
        # Calcul des 20 deltas action-to-action
        # delta[0] = a[i] - a[i-1] ; delta[j] = a[i+j] - a[i+j-1] pour j ≥ 1
        chunk_actions = actions[i - 1: i + CHUNK_SIZE]  # 21 actions de i-1 à i+19
        deltas = chunk_actions[1:] - chunk_actions[:-1]  # 20 deltas
        valid_feats.append(features[i])
        valid_deltas.append(deltas)
        i += 1
    X_feat = torch.stack(valid_feats)
    D = torch.stack(valid_deltas)  # (N, 20, 2) en espace action normalisé
    print(f"  {len(X_feat)} samples (chunk={CHUNK_SIZE}, deltas action→action)", flush=True)
    return X_feat, D


def build_uniform_grid(train_deltas, n_per_axis, percentile):
    """Construit une grille uniforme n×n centrée sur 0, bornée par le percentile.

    train_deltas : (N, 20, 2)
    Retourne : bin_centers (n², 2) en même espace que train_deltas.
    """
    flat = train_deltas.reshape(-1, 2)
    # Bornes : ± percentile en valeur absolue
    bound_x = float(torch.quantile(flat[:, 0].abs(), percentile / 100))
    bound_y = float(torch.quantile(flat[:, 1].abs(), percentile / 100))
    bound = max(bound_x, bound_y)  # même borne sur les 2 axes (symétrie)
    print(f"  Bornes grille (P{percentile}%) : ±{bound:.4f} (norm) ≈ ±?? px", flush=True)

    axis = torch.linspace(-bound, bound, n_per_axis)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (n², 2)
    print(f"  Grille {n_per_axis}×{n_per_axis} = {centers.shape[0]} bins", flush=True)
    return centers, bound


def assign_bins(deltas, centers):
    """Pour chaque delta, trouve l'index du bin le plus proche."""
    flat = deltas.reshape(-1, 2)
    dists = torch.cdist(flat, centers)
    idx = dists.argmin(dim=-1)
    return idx.reshape(deltas.shape[:-1])


class TransformerDeltaGrid(nn.Module):
    """Transformer image-only, sortie discrète sur grille de deltas + résidu."""

    def __init__(self, feature_dim, d_model, n_heads, n_layers, dim_feedforward,
                 chunk_size, n_bins, bin_centers, y_std, action_dim=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = n_bins

        # bin_centers en espace delta normalisé (= même espace que la target d'entraînement)
        self.register_buffer("bin_centers", bin_centers)
        # y_std : pour dénormaliser les deltas en pixels à l'inférence
        self.register_buffer("y_std", y_std)

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

    def compute_loss(self, bin_logits, residuals, target_deltas):
        """target_deltas: (batch, chunk, 2) en espace delta normalisé."""
        batch, chunk, n_bins = bin_logits.shape
        target_bins = assign_bins(target_deltas, self.bin_centers)
        ce_loss = F.cross_entropy(
            bin_logits.reshape(-1, n_bins),
            target_bins.reshape(-1),
        )
        target_residuals = target_deltas - self.bin_centers[target_bins]
        idx = target_bins.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        sel_residual = torch.gather(residuals, dim=2, index=idx).squeeze(2)
        mse_loss = F.mse_loss(sel_residual, target_residuals)
        return ce_loss + RESIDUAL_WEIGHT * mse_loss, ce_loss.item(), mse_loss.item()

    def predict_deltas(self, bin_logits, residuals, mode='sample'):
        """Retourne les 20 deltas prédits, en espace delta normalisé."""
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


def train_model(model, X_feat, D, train_idx, test_idx, norm):
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
            loss, ce_v, mse_v = model.compute_loss(bin_logits, residuals, D[batch])
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
            test_loss, _, _ = model.compute_loss(bin_logits, residuals, D[test_idx])
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

    # y_std en tensor pour denormalisation (déjà sur le device CPU)
    y_std = norm["y_std"].numpy() if isinstance(norm["y_std"], torch.Tensor) else np.array(norm["y_std"])

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        max_coverage = 0
        # Bootstrap : au démarrage, "dernière action commandée" = agent_pos
        last_commanded = obs["agent_pos"].astype(np.float32).copy()
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
                    pred_deltas_norm = model.predict_deltas(bin_logits, residuals, mode=eval_mode)
                # Dénormaliser les deltas en pixels
                pred_deltas_pix = (pred_deltas_norm.squeeze(0) * y_std).numpy()  # (20, 2)
                # Reconstruction cumulative depuis last_commanded
                actions_chunk = []
                prev = last_commanded.copy()
                for j in range(CHUNK_SIZE):
                    next_action = prev + pred_deltas_pix[j]
                    actions_chunk.append(next_action.copy())
                    prev = next_action
                action_buffer = actions_chunk

            action = np.clip(action_buffer.pop(0), 0, 512).astype(np.float32)
            last_commanded = action.copy()  # mémoriser pour le prochain chunk
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
    label = "resnet18_delta_grid"
    arch_name = f"ResNet18(gelé) → Transformer 2L → Grille delta {N_BINS_PER_AXIS}×{N_BINS_PER_AXIS} + résidu chunk={CHUNK_SIZE}"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nFeatures ResNet18...", flush=True)
    features, positions, actions, norm, episodes = extract_features()

    print("\nConstruction des chunks (deltas action→action)...", flush=True)
    X_feat, D = load_data_chunked_delta(features, actions, episodes)

    # Split train/test
    n = len(X_feat)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    # Stats des deltas train (info + base pour la grille)
    train_deltas = D[train_idx]  # (N_train, 20, 2)
    flat_train = train_deltas.reshape(-1, 2)
    d_std = flat_train.std(dim=0)
    d_mean = flat_train.mean(dim=0)
    print(f"  Deltas train : mean={d_mean.tolist()} std={d_std.tolist()} (espace action normalisé)", flush=True)
    print(f"  Équivalent pixels : std≈({d_std[0]*norm['y_std'][0]:.2f}, {d_std[1]*norm['y_std'][1]:.2f})", flush=True)

    # Construction de la grille uniforme
    print("\nConstruction de la grille uniforme...", flush=True)
    bin_centers, grid_bound = build_uniform_grid(train_deltas, N_BINS_PER_AXIS, GRID_PERCENTILE)
    print(f"  Équivalent grille en pixels : ±{grid_bound * norm['y_std'].max().item():.1f} px", flush=True)

    # y_std en tensor pour le buffer du modèle
    y_std_tensor = norm["y_std"] if isinstance(norm["y_std"], torch.Tensor) else torch.tensor(norm["y_std"], dtype=torch.float32)

    model = TransformerDeltaGrid(
        feature_dim=FEATURE_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        chunk_size=CHUNK_SIZE, n_bins=N_BINS,
        bin_centers=bin_centers, y_std=y_std_tensor,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)

    print(f"\nEntraînement (grille delta {N_BINS_PER_AXIS}×{N_BINS_PER_AXIS}, image seule)...", flush=True)
    (train_losses, test_losses, ce_train, mse_train, epoch_times,
     train_time, n_train, n_test, checkpoint_evals,
     best_state_dict, best_coverage, best_epoch) = train_model(
        model, X_feat, D, train_idx, test_idx, norm
    )

    # Inférence timing (transformer + sample seulement, sans simu)
    dummy_feat = torch.randn(1, FEATURE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            bin_logits, residuals = model(dummy_feat)
            model.predict_deltas(bin_logits, residuals, mode='sample')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            bin_logits, residuals = model(dummy_feat)
            model.predict_deltas(bin_logits, residuals, mode='sample')
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale (mode=sample, capture la multimodalité)...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, norm, eval_mode='sample', save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "TransformerDeltaGrid",
        "feature_dim": FEATURE_DIM, "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
        "chunk_size": CHUNK_SIZE, "n_bins": N_BINS,
        "n_bins_per_axis": N_BINS_PER_AXIS,
        "grid_percentile": GRID_PERCENTILE,
        "grid_bound_normalized": float(grid_bound),
        "note": ("Grille uniforme {0}×{0} sur deltas humains action→action. "
                 "Reconstruction cumulative en pixels à l'inférence. "
                 "Bootstrap: agent_pos au step 0 d'épisode.").format(N_BINS_PER_AXIS),
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

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
            "loss_type": "cross_entropy_plus_mse_residual_on_deltas",
            "n_bins": N_BINS,
            "n_bins_per_axis": N_BINS_PER_AXIS,
            "grid_percentile": GRID_PERCENTILE,
            "grid_bound_normalized": float(grid_bound),
            "residual_weight": RESIDUAL_WEIGHT,
            "ce_train_per_epoch": ce_train,
            "mse_train_per_epoch": mse_train,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch,
            "best_checkpoint_coverage": best_coverage,
            "checkpoint_eval_mode": "argmax",
            "final_eval_mode": "sample",
            "description": ("Bins en grille uniforme 8×8 sur deltas humains action→action. "
                            "Image seule, reconstruction cumulative."),
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
    print(f"\n  Checkpoint evals (argmax) :", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} "
              f"(ce={cp['ce_loss']:.4f} mse={cp['mse_loss']:.4f}) "
              f"test={cp['test_loss']:.4f} cov={cp['coverage']:.1%}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
