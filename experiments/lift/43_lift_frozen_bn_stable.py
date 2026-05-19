"""
Expérience 43 — Lift : Run 42 + fix bug NaN.

Run 42 a atteint 66% success rate à epoch 25 (best result du projet), MAIS
les poids ont divergé en NaN après ça (BN trainable + MPS + AdamW instable).

3 fixes vs run 42 :
1. **FrozenBatchNorm2d** (torchvision.ops.misc) : remplace les BN par une
   version aux stats constantes, pas de gradient possible → évite le mode
   train/eval problème + évite l'explosion des BN.
2. **NaN guard** : si loss devient NaN/inf, on skip backward+step au lieu de
   propager. Empêche la chaîne en cascade.
3. **Save best checkpoint à disque IMMÉDIATEMENT** quand on atteint un nouveau
   best (pas juste à la fin). Sécurise si crash NaN ultérieur.

Reste identique : ResNet trainable (sauf BN figées), 2 LR groups (1e-5/1e-3),
anti-overfit combo (Dropout 0.4 + LayerNorm + AdamW wd=5e-4 + label_smoothing 0.1),
image+state+CE+résidu, MPS.
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.lift_image_features import extract_lift_raw_images
from src.lift_data import build_state_vector, STATE_KEYS
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "lift/43_lift_frozen_bn_stable"
DATASET = "robomimic/lift_ph"

FEATURE_DIM = 512
STATE_DIM = 19
ACTION_DIM = 7
CHUNK_SIZE = 20
IMAGE_SIZE = 96

HIDDEN_DIMS = [256, 256]
N_BINS = 21
RESIDUAL_WEIGHT = 1.0

# Anti-overfit (idem run 40)
DROPOUT = 0.4
WEIGHT_DECAY_HEAD = 5e-4
WEIGHT_DECAY_BACKBONE = 1e-4
LABEL_SMOOTHING = 0.1
USE_LAYER_NORM = True

# LR groups
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5         # ↓ ultra bas pour pas casser les features ImageNet

WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

N_EVAL_EPISODES_CHECKPOINT = 50
N_EVAL_EPISODES_FINAL = 200
MAX_STEPS = 200
CHECKPOINT_EPOCHS = [25, 50, 75, 100]

# Device : Apple Silicon GPU (MPS) si dispo
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Normalisation ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# ============================================================


def build_bin_centers(n_bins):
    edges = torch.linspace(-1.0, 1.0, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def assign_bins(actions, n_bins):
    bin_width = 2.0 / n_bins
    return ((actions + 1.0) / bin_width).long().clamp(0, n_bins - 1)


class ImageStateE2E(nn.Module):
    """ResNet18 trainable (BN gelées) + MLP head → CE+résidu par dim d'action."""

    def __init__(self, state_dim, hidden_dims, chunk_size, action_dim, n_bins,
                 state_mean, state_std, dropout=DROPOUT, use_layer_norm=USE_LAYER_NORM,
                 label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.label_smoothing = label_smoothing

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("bin_centers", build_bin_centers(n_bins))
        self.register_buffer("imagenet_mean", IMAGENET_MEAN)
        self.register_buffer("imagenet_std", IMAGENET_STD)

        # ResNet18 (pré-entraîné ImageNet) avec BN remplacé par FrozenBatchNorm2d
        # (recette ACT LeRobot : stats figées, pas de mode train/eval qui pose souci sur MPS)
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.ops.misc import FrozenBatchNorm2d
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        def freeze_bn(module):
            """Remplace récursivement BatchNorm2d par FrozenBatchNorm2d."""
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    new_bn = FrozenBatchNorm2d(child.num_features, eps=child.eps)
                    new_bn.weight.data = child.weight.data.clone()
                    new_bn.bias.data = child.bias.data.clone()
                    new_bn.running_mean.data = child.running_mean.data.clone()
                    new_bn.running_var.data = child.running_var.data.clone()
                    setattr(module, name, new_bn)
                else:
                    freeze_bn(child)
        freeze_bn(backbone)
        self.backbone = backbone

        # MLP head (idem run 40)
        in_dim = FEATURE_DIM + state_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.bin_head = nn.Linear(prev, chunk_size * action_dim * n_bins)
        self.residual_head = nn.Linear(prev, chunk_size * action_dim * n_bins)

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def normalize_image(self, image_uint8):
        """image_uint8 : (B, 3, H, W) uint8 — retourne (B, 3, H, W) float normalisé."""
        img = image_uint8.float() / 255.0
        return (img - self.imagenet_mean) / self.imagenet_std

    def forward(self, image, state):
        # Normalisation hors-graphe pour éviter les soucis MPS de broadcast view
        with torch.no_grad():
            img_norm = self.normalize_image(image).clone()
        img_norm.requires_grad_(False)
        features_raw = self.backbone(img_norm)
        features = features_raw.reshape(image.shape[0], -1).contiguous()
        state_norm = self.normalize_state(state)
        x = torch.cat([features, state_norm], dim=-1)
        h = self.body(x)
        bin_logits = self.bin_head(h).reshape(-1, self.chunk_size, self.action_dim, self.n_bins).contiguous()
        residuals = self.residual_head(h).reshape(-1, self.chunk_size, self.action_dim, self.n_bins).contiguous()
        return bin_logits, residuals

    def compute_loss(self, bin_logits, residuals, target_actions):
        N = self.n_bins
        target_bins = assign_bins(target_actions, N)
        ce_loss = F.cross_entropy(
            bin_logits.reshape(-1, N), target_bins.reshape(-1),
            label_smoothing=self.label_smoothing,
        )
        target_residuals = target_actions - self.bin_centers[target_bins]
        idx = target_bins.unsqueeze(-1)
        sel_residual = torch.gather(residuals, dim=-1, index=idx).squeeze(-1)
        mse_loss = F.mse_loss(sel_residual, target_residuals)
        return ce_loss + RESIDUAL_WEIGHT * mse_loss, ce_loss.item(), mse_loss.item()

    def predict(self, bin_logits, residuals, mode='argmax'):
        if mode == 'argmax':
            bin_ids = bin_logits.argmax(dim=-1)
        else:
            probs = F.softmax(bin_logits, dim=-1)
            bin_ids = torch.multinomial(probs.reshape(-1, self.n_bins), 1).reshape(*probs.shape[:-1])
        centers = self.bin_centers[bin_ids]
        idx = bin_ids.unsqueeze(-1)
        sel_residual = torch.gather(residuals, dim=-1, index=idx).squeeze(-1)
        return (centers + sel_residual).clamp(-1.0, 1.0)


def build_chunks(images, states, actions, episodes, chunk_size):
    """images : (N, 3, H, W) uint8. On garde l'index t0, et le chunk d'actions."""
    valid_img_idx, valid_states, valid_chunks, chunk_eps = [], [], [], []
    n = len(images)
    i = 0
    while i + chunk_size <= n:
        if episodes[i] == episodes[i + chunk_size - 1]:
            valid_img_idx.append(i)
            valid_states.append(states[i])
            valid_chunks.append(actions[i:i + chunk_size])
            chunk_eps.append(int(episodes[i]))
            i += 1
        else:
            i += 1
    idx_tensor = torch.tensor(valid_img_idx, dtype=torch.long)
    XS = torch.stack(valid_states)
    Y = torch.stack(valid_chunks)
    print(f"  {len(idx_tensor)} chunks (chunk_size={chunk_size})", flush=True)
    return idx_tensor, XS, Y, np.array(chunk_eps, dtype=np.int64)


def split_by_episode(chunk_episodes, n_demos, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    all_eps = np.arange(n_demos)
    rng.shuffle(all_eps)
    n_train_eps = int(train_ratio * n_demos)
    train_eps = set(all_eps[:n_train_eps].tolist())
    test_eps = set(all_eps[n_train_eps:].tolist())
    train_mask = np.array([e in train_eps for e in chunk_episodes])
    test_mask = np.array([e in test_eps for e in chunk_episodes])
    return (torch.tensor(np.where(train_mask)[0], dtype=torch.long),
            torch.tensor(np.where(test_mask)[0], dtype=torch.long),
            len(train_eps), len(test_eps))


def train_model(model, images, img_idx, XS, Y, train_idx, test_idx, run_dir=None):
    torch.manual_seed(SEED)

    # 2 LR groups
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY_HEAD},
    ])
    n_bb = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"  Backbone trainable : {n_bb:,} params (LR={LR_BACKBONE})", flush=True)
    print(f"  Head     trainable : {n_head:,} params (LR={LR_HEAD})", flush=True)

    train_losses, test_losses = [], []
    ce_train, mse_train = [], []
    epoch_times = []
    checkpoint_evals = []
    best_success = -1.0
    best_state_dict = None
    best_epoch = None
    start_total = time.time()

    for epoch in range(EPOCHS):
        # Warmup
        if epoch < WARMUP_EPOCHS:
            for pg, base_lr in zip(optimizer.param_groups, [LR_BACKBONE, LR_HEAD]):
                pg["lr"] = base_lr * (epoch + 1) / WARMUP_EPOCHS
        else:
            for pg, base_lr in zip(optimizer.param_groups, [LR_BACKBONE, LR_HEAD]):
                pg["lr"] = base_lr

        start_ep = time.time()
        model.train()
        epoch_loss, epoch_ce, epoch_mse, n_batches = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(len(train_idx))

        n_skipped = 0
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            img_batch = images[img_idx[batch]].contiguous().to(DEVICE, non_blocking=True)
            state_batch = XS[batch].contiguous().to(DEVICE, non_blocking=True)
            y_batch = Y[batch].contiguous().to(DEVICE, non_blocking=True)
            bin_logits, residuals = model(img_batch, state_batch)
            loss, ce_v, mse_v = model.compute_loss(bin_logits, residuals, y_batch)
            # NaN guard : skip si loss explose
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item(); epoch_ce += ce_v; epoch_mse += mse_v; n_batches += 1
        if n_skipped > 0:
            print(f"  [WARN epoch {epoch+1}] {n_skipped} batches skipped (NaN/inf loss)", flush=True)

        avg_train = epoch_loss / n_batches
        avg_ce = epoch_ce / n_batches
        avg_mse = epoch_mse / n_batches

        # Test : par batches pour ne pas exploser la mémoire
        model.eval()
        test_losses_batch = []
        with torch.no_grad():
            for i in range(0, len(test_idx), BATCH_SIZE):
                tbatch = test_idx[i:i + BATCH_SIZE]
                img_b = images[img_idx[tbatch]].contiguous().to(DEVICE, non_blocking=True)
                state_b = XS[tbatch].contiguous().to(DEVICE, non_blocking=True)
                y_b = Y[tbatch].contiguous().to(DEVICE, non_blocking=True)
                bin_logits, residuals = model(img_b, state_b)
                tloss, _, _ = model.compute_loss(bin_logits, residuals, y_b)
                test_losses_batch.append(tloss.item() * len(tbatch))
        test_loss = sum(test_losses_batch) / len(test_idx)

        ep_time = time.time() - start_ep
        train_losses.append(avg_train); test_losses.append(test_loss)
        ce_train.append(avg_ce); mse_train.append(avg_mse)
        epoch_times.append(ep_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train: {avg_train:.4f} "
                  f"(ce={avg_ce:.4f} mse={avg_mse:.4f}) test: {test_loss:.4f} — {elapsed:.0f}s",
                  flush=True)

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            print(f"\n  [CHECKPOINT] Éval à epoch {epoch+1} (argmax)...", flush=True)
            eval_res, _ = eval_in_simulation(model, eval_mode='argmax', save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1, "train_loss": avg_train, "test_loss": test_loss,
                "ce_loss": avg_ce, "mse_loss": avg_mse,
                "success_rate": eval_res["success_rate"],
                "avg_max_cube_z": eval_res["avg_max_cube_z"],
            })
            print(f"  [CHECKPOINT epoch {epoch+1}] train={avg_train:.4f} test={test_loss:.4f} "
                  f"success={eval_res['success_rate']:.0%} max_z={eval_res['avg_max_cube_z']:.3f}",
                  flush=True)
            if eval_res["success_rate"] > best_success:
                best_success = eval_res["success_rate"]
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"  [NEW BEST] epoch {epoch+1} : success {best_success:.0%}", flush=True)
                # Save IMMÉDIATEMENT à disque (sécurité contre NaN cascade)
                if run_dir is not None:
                    best_path = run_dir / f"model_best_epoch{best_epoch}.pt"
                    torch.save({"state_dict": best_state_dict, "epoch": best_epoch,
                                "success_rate": best_success}, best_path)
                    print(f"  → sauvegardé à {best_path.name}", flush=True)

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s", flush=True)
    return (train_losses, test_losses, ce_train, mse_train, epoch_times, total_time,
            len(train_idx), len(test_idx), checkpoint_evals,
            best_state_dict, best_success, best_epoch)


def eval_in_simulation(model, eval_mode='argmax', save_videos=True,
                       n_eval_episodes=N_EVAL_EPISODES_CHECKPOINT):
    import robosuite as rs
    print(f"  Évaluation ({n_eval_episodes} épisodes, mode={eval_mode})...", flush=True)

    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )

    results = []
    all_frames = []
    model.eval()

    for ep in range(n_eval_episodes):
        obs = env.reset()
        frames = []
        ep_reward = 0
        max_cube_z = 0
        success = False
        action_buffer = []

        for step in range(MAX_STEPS):
            if save_videos and ep < 3:
                frame = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
                frames.append(frame)

            if len(action_buffer) == 0:
                img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
                # uint8 (1, 3, H, W) → device
                img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                state_np = build_state_vector(obs).astype(np.float32)
                state_tensor = torch.tensor(state_np).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    bin_logits, residuals = model(img_t, state_tensor)
                    pred_actions = model.predict(bin_logits, residuals, mode=eval_mode).squeeze(0).cpu().numpy()
                action_buffer = list(pred_actions)

            action = action_buffer.pop(0)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if "cube_pos" in obs:
                max_cube_z = max(max_cube_z, obs["cube_pos"][2])
            if env._check_success():
                success = True
            if done:
                break

        results.append({"reward": ep_reward, "success": success, "max_cube_z": max_cube_z})
        if (ep + 1) % max(1, n_eval_episodes // 5) == 0:
            status = "✓" if success else "✗"
            print(f"    Episode {ep+1:3d}/{n_eval_episodes} : {status} max_z={max_cube_z:.3f}",
                  flush=True)
        if frames:
            all_frames.append(frames)

    env.close()
    n_success = sum(r["success"] for r in results)
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_max_z = float(np.mean([r["max_cube_z"] for r in results]))
    print(f"    Success rate : {n_success}/{n_eval_episodes} "
          f"({n_success/n_eval_episodes:.0%}) — avg max cube_z : {avg_max_z:.3f} m",
          flush=True)
    return {
        "success_rate": n_success / n_eval_episodes,
        "avg_reward": avg_reward,
        "avg_max_cube_z": avg_max_z,
    }, all_frames


def main():
    label = "mlp_unfrozen_resnet"
    arch_name = (f"ResNet18(trainable, BN gelée, lr={LR_BACKBONE}) + MLP "
                 f"{FEATURE_DIM+STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} "
                 f"(LN,DO={DROPOUT}) → CE+résidu (LS={LABEL_SMOOTHING}, head_lr={LR_HEAD})")

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nChargement images brutes + states...", flush=True)
    data = extract_lift_raw_images(image_size=IMAGE_SIZE)
    images = data["images"]                # (N, 3, H, W) uint8
    states = data["states"]
    actions = data["actions"]
    episodes = data["episodes"]
    n_demos = data["meta"]["n_demos"]
    print(f"  Images shape : {tuple(images.shape)} (dtype={images.dtype})", flush=True)

    print("\nConstruction des chunks...", flush=True)
    img_idx, XS, Y, chunk_eps = build_chunks(images, states, actions, episodes, CHUNK_SIZE)
    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(
        chunk_eps, n_demos, train_ratio=0.8, seed=SEED
    )
    print(f"  Split : {n_train_eps} démos train ({len(train_idx)} chunks), "
          f"{n_test_eps} démos test ({len(test_idx)} chunks)", flush=True)

    state_mean = XS[train_idx].mean(dim=0)
    state_std = XS[train_idx].std(dim=0).clamp(min=1e-6)

    model = ImageStateE2E(
        state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM, n_bins=N_BINS,
        state_mean=state_mean, state_std=state_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Total params : {info['total_params']:,} (dont ResNet18 ~11.2M trainable)",
          flush=True)
    print(f"  Device : {DEVICE}", flush=True)
    model = model.to(DEVICE)

    # Pré-création du run_dir pour pouvoir sauver best en cours de training
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    print(f"\n  Run dir : {run_dir}", flush=True)

    print("\nEntraînement (ResNet trainable, 2 LR groups)...", flush=True)
    (train_losses, test_losses, ce_train, mse_train, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(
        model, images, img_idx, XS, Y, train_idx, test_idx, run_dir
    )

    # Inference timing : 1 image dummy
    dummy_img = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8).to(DEVICE)
    dummy_state = torch.randn(1, STATE_DIM).to(DEVICE)
    model.eval()
    for _ in range(10):
        with torch.no_grad():
            bl, rs_ = model(dummy_img, dummy_state)
            model.predict(bl, rs_, mode='argmax')
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad():
            bl, rs_ = model(dummy_img, dummy_state)
            model.predict(bl, rs_, mode='argmax')
    inf_time = (time.time() - t0) / 50 * 1000

    print("\nÉval finale 200 eps (argmax + vidéos)...", flush=True)
    eval_results, all_frames = eval_in_simulation(
        model, eval_mode='argmax', save_videos=True,
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

    # run_dir déjà créé avant training
    model_config = {
        "class": "ImageStateE2E",
        "state_dim": STATE_DIM, "hidden_dims": HIDDEN_DIMS,
        "chunk_size": CHUNK_SIZE, "action_dim": ACTION_DIM, "n_bins": N_BINS,
        "image_size": IMAGE_SIZE,
        "backbone": "resnet18 ImageNet, BN frozen, rest trainable",
        "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
        "dropout": DROPOUT, "label_smoothing": LABEL_SMOOTHING,
        "weight_decay_backbone": WEIGHT_DECAY_BACKBONE,
        "weight_decay_head": WEIGHT_DECAY_HEAD,
        "note": "Run 40 + ResNet18 dégelé (lr=1e-5). LeRobot-style.",
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=20)
    if best_state_dict is not None:
        best_path = run_dir / f"model_best_epoch{best_epoch}.pt"
        torch.save({"state_dict": best_state_dict, "epoch": best_epoch,
                    "success_rate": best_success, "config": model_config}, best_path)
        print(f"  Best model : {best_path} (epoch {best_epoch}, success {best_success:.0%})",
              flush=True)

    batches_per_epoch = n_train // BATCH_SIZE + 1
    flops_inf = sum(2 * p.numel() for p in model.parameters())

    run_data = build_run_data(
        experiment=EXPERIMENT_NAME, architecture=arch_name, dataset=DATASET,
        n_episodes=n_demos, n_params=info["total_params"], size_mb=info["size_mb"],
        seed=SEED, train_losses=train_losses, test_losses=test_losses,
        train_time_s=train_time, epoch_times=epoch_times,
        flops_total_train=flops_inf * BATCH_SIZE * batches_per_epoch * EPOCHS,
        n_train_samples=n_train, n_test_samples=n_test,
        inference_time_ms=inf_time, flops_inference=flops_inf,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_max_cube_z"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "task": "Robomimic Lift (Panda 7-DoF + cube)",
            "feature_extractor": "ResNet18 (trainable, BN frozen, lr=1e-5)",
            "image_size": IMAGE_SIZE,
            "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
            "n_demos": n_demos,
            "n_train_episodes": n_train_eps, "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch, "best_checkpoint_success": best_success,
            "ce_train_per_epoch": ce_train, "mse_train_per_epoch": mse_train,
            "loss_type": "cross_entropy_per_dim_plus_mse_residual",
            "description": "ResNet18 fine-tuné (lr=1e-5) + anti-overfit (DO 0.4, LN, AdamW, LS 0.1). LeRobot-style.",
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay_head": WEIGHT_DECAY_HEAD,
            "weight_decay_backbone": WEIGHT_DECAY_BACKBONE,
            "n_eval_episodes_final": N_EVAL_EPISODES_FINAL,
            "n_eval_episodes_checkpoint": N_EVAL_EPISODES_CHECKPOINT,
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"RÉSUMÉ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Success rate final : {eval_results['success_rate']:.0%} "
          f"({int(eval_results['success_rate']*N_EVAL_EPISODES_FINAL)}/{N_EVAL_EPISODES_FINAL})",
          flush=True)
    print(f"  Best checkpoint    : {best_success:.0%} (epoch {best_epoch})", flush=True)
    print(f"  Avg max cube_z     : {eval_results['avg_max_cube_z']:.3f} m", flush=True)
    print(f"  Params             : {info['total_params']:,}", flush=True)
    print(f"  Temps              : {train_time:.0f}s", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} test={cp['test_loss']:.4f} "
              f"success={cp['success_rate']:.0%} max_z={cp['avg_max_cube_z']:.3f}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
