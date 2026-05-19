"""
Expérience 39 — Lift : image aug avec 50/50 v0/aug (fix le distribution shift).

Run 38 augmentait toujours (random parmi 8 versions, dont 1 originale = v0).
→ 7/8 = 87.5% des samples étaient augmentés → distribution train ≠ eval.
→ Résultat : 20% best, pire que run 36 (34%, no aug).

Correction : 50% des samples = v0 (original, matche eval), 50% = v1..v7 (aug).
→ Le modèle voit toujours la distribution éval clean, mais bénéficie de la
  diversité d'aug pour réduire l'overfit.

Tout le reste identique au run 38.
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
from src.lift_image_features import extract_lift_image_features_augmented
from src.lift_data import build_state_vector, STATE_KEYS
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "lift/39_lift_image_aug_50pct"
DATASET = "robomimic/lift_ph"

FEATURE_DIM = 512
STATE_DIM = 19
ACTION_DIM = 7
CHUNK_SIZE = 20
IMAGE_SIZE = 96
N_VERSIONS = 8                # nombre de versions augmentées par frame

HIDDEN_DIMS = [256, 256]
N_BINS = 21
RESIDUAL_WEIGHT = 1.0

LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

N_EVAL_EPISODES = 50
MAX_STEPS = 200
CHECKPOINT_EPOCHS = [25, 50, 75, 100]
# ============================================================


def build_bin_centers(n_bins):
    edges = torch.linspace(-1.0, 1.0, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def assign_bins(actions, n_bins):
    bin_width = 2.0 / n_bins
    return ((actions + 1.0) / bin_width).long().clamp(0, n_bins - 1)


class MLPImageStateDiscrete(nn.Module):
    """Identique à run 36 — MLP (features+state) → CE+résidu par dim."""

    def __init__(self, feature_dim, state_dim, hidden_dims, chunk_size, action_dim, n_bins,
                 state_mean, state_std):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = n_bins

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("bin_centers", build_bin_centers(n_bins))

        in_dim = feature_dim + state_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.body = nn.Sequential(*layers)
        self.bin_head = nn.Linear(prev, chunk_size * action_dim * n_bins)
        self.residual_head = nn.Linear(prev, chunk_size * action_dim * n_bins)

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def forward(self, features, state):
        x = torch.cat([features, state], dim=-1)
        h = self.body(x)
        bin_logits = self.bin_head(h).view(-1, self.chunk_size, self.action_dim, self.n_bins)
        residuals = self.residual_head(h).view(-1, self.chunk_size, self.action_dim, self.n_bins)
        return bin_logits, residuals

    def compute_loss(self, bin_logits, residuals, target_actions):
        N = self.n_bins
        target_bins = assign_bins(target_actions, N)
        ce_loss = F.cross_entropy(bin_logits.reshape(-1, N), target_bins.reshape(-1))
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


def build_chunks_aug(features_aug, states, actions, episodes, chunk_size):
    """Construit chunks. Conserve les n_versions par frame de départ.

    features_aug : (N_frames, n_versions, 512) — n versions par frame.
    Retourne XF_aug (n_chunks, n_versions, 512), XS, Y, chunk_eps.
    """
    valid_feats, valid_states, valid_chunks, chunk_eps = [], [], [], []
    n = len(features_aug)
    i = 0
    while i + chunk_size <= n:
        if episodes[i] == episodes[i + chunk_size - 1]:
            valid_feats.append(features_aug[i])         # (n_versions, 512)
            valid_states.append(states[i])
            valid_chunks.append(actions[i:i + chunk_size])
            chunk_eps.append(int(episodes[i]))
            i += 1
        else:
            i += 1
    XF = torch.stack(valid_feats)        # (n_chunks, n_versions, 512)
    XS = torch.stack(valid_states)
    Y = torch.stack(valid_chunks)
    print(f"  {len(XF)} chunks (chunk_size={chunk_size}), {XF.shape[1]} versions par chunk", flush=True)
    return XF, XS, Y, np.array(chunk_eps, dtype=np.int64)


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


def train_model(model, XF, XS, Y, train_idx, test_idx, n_versions):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(SEED)

    train_losses, test_losses = [], []
    ce_train, mse_train = [], []
    epoch_times = []
    checkpoint_evals = []
    best_success = -1.0
    best_state_dict = None
    best_epoch = None
    start_total = time.time()

    for epoch in range(EPOCHS):
        lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else LEARNING_RATE
        for pg in optimizer.param_groups: pg["lr"] = lr

        start_ep = time.time()
        model.train()
        epoch_loss, epoch_ce, epoch_mse, n_batches = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            # 50% v0 (original, matche eval), 50% v1..v_{n-1} (augmenté)
            use_v0 = torch.rand(len(batch), generator=gen) < 0.5
            random_aug = torch.randint(1, n_versions, (len(batch),), generator=gen)
            ver_ids = torch.where(use_v0, torch.zeros_like(random_aug), random_aug)
            XF_batch = XF[batch, ver_ids]   # (B, 512)
            state_in = model.normalize_state(XS[batch])
            bin_logits, residuals = model(XF_batch, state_in)
            loss, ce_v, mse_v = model.compute_loss(bin_logits, residuals, Y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item(); epoch_ce += ce_v; epoch_mse += mse_v; n_batches += 1

        avg_train = epoch_loss / n_batches
        avg_ce = epoch_ce / n_batches
        avg_mse = epoch_mse / n_batches

        # Test : sur la version v0 (originale) — comme à l'éval
        model.eval()
        with torch.no_grad():
            state_in = model.normalize_state(XS[test_idx])
            bin_logits, residuals = model(XF[test_idx, 0], state_in)
            test_loss, _, _ = model.compute_loss(bin_logits, residuals, Y[test_idx])
            test_loss = test_loss.item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train); test_losses.append(test_loss)
        ce_train.append(avg_ce); mse_train.append(avg_mse)
        epoch_times.append(ep_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train: {avg_train:.4f} "
                  f"(ce={avg_ce:.4f} mse={avg_mse:.4f}) test: {test_loss:.4f} — {elapsed:.0f}s", flush=True)

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
                  f"success={eval_res['success_rate']:.0%} max_z={eval_res['avg_max_cube_z']:.3f}", flush=True)
            if eval_res["success_rate"] > best_success:
                best_success = eval_res["success_rate"]
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"  [NEW BEST] epoch {epoch+1} : success {best_success:.0%}", flush=True)

    total_time = time.time() - start_total
    print(f"  Temps total : {total_time:.1f}s", flush=True)
    return (train_losses, test_losses, ce_train, mse_train, epoch_times, total_time,
            len(train_idx), len(test_idx), checkpoint_evals,
            best_state_dict, best_success, best_epoch)


def eval_in_simulation(model, eval_mode='argmax', save_videos=True):
    import robosuite as rs
    from torchvision.models import resnet18, ResNet18_Weights

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes, mode={eval_mode})...", flush=True)

    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    results = []
    all_frames = []
    model.eval()

    for ep in range(N_EVAL_EPISODES):
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
                img_t = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_t = (img_t - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    features = backbone(img_t).flatten(1)
                state_np = build_state_vector(obs).astype(np.float32)
                state_tensor = torch.tensor(state_np).unsqueeze(0)
                state_norm = model.normalize_state(state_tensor)
                with torch.no_grad():
                    bin_logits, residuals = model(features, state_norm)
                    pred_actions = model.predict(bin_logits, residuals, mode=eval_mode).squeeze(0).numpy()
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
        if (ep + 1) % 10 == 0:
            status = "✓" if success else "✗"
            print(f"    Episode {ep+1:2d}/{N_EVAL_EPISODES} : {status} max_z={max_cube_z:.3f}", flush=True)
        if frames:
            all_frames.append(frames)

    env.close()
    n_success = sum(r["success"] for r in results)
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_max_z = float(np.mean([r["max_cube_z"] for r in results]))
    print(f"    Success rate : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%}) "
          f"— avg max cube_z : {avg_max_z:.3f} m", flush=True)
    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": avg_reward,
        "avg_max_cube_z": avg_max_z,
    }, all_frames


def main():
    label = "mlp_feat_state_ce_aug50pct"
    arch_name = f"ResNet18(gelé,aug50%) + MLP {FEATURE_DIM+STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → CE+résidu (n_aug={N_VERSIONS})"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print(f"\nChargement features augmentées (n_versions={N_VERSIONS})...", flush=True)
    data = extract_lift_image_features_augmented(image_size=IMAGE_SIZE, n_versions=N_VERSIONS)
    features_aug = data["features"]                # (N, n_versions, 512)
    states = data["states"]; actions = data["actions"]; episodes = data["episodes"]
    n_demos = data["meta"]["n_demos"]
    print(f"  Features augmentées shape : {features_aug.shape}", flush=True)

    print("\nConstruction des chunks...", flush=True)
    XF, XS, Y, chunk_eps = build_chunks_aug(features_aug, states, actions, episodes, CHUNK_SIZE)
    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(chunk_eps, n_demos, train_ratio=0.8, seed=SEED)
    print(f"  Split : {n_train_eps} démos train ({len(train_idx)} chunks), {n_test_eps} démos test ({len(test_idx)} chunks)", flush=True)

    state_mean = XS[train_idx].mean(dim=0)
    state_std = XS[train_idx].std(dim=0).clamp(min=1e-6)

    model = MLPImageStateDiscrete(
        feature_dim=FEATURE_DIM, state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM, n_bins=N_BINS,
        state_mean=state_mean, state_std=state_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)

    print("\nEntraînement (avec random pick augmenté par sample)...", flush=True)
    (train_losses, test_losses, ce_train, mse_train, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(
        model, XF, XS, Y, train_idx, test_idx, N_VERSIONS)

    dummy_feat = torch.randn(1, FEATURE_DIM)
    dummy_state = torch.randn(1, STATE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            bl, rs_ = model(dummy_feat, model.normalize_state(dummy_state))
            model.predict(bl, rs_, mode='argmax')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            bl, rs_ = model(dummy_feat, model.normalize_state(dummy_state))
            model.predict(bl, rs_, mode='argmax')
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale (argmax + vidéos)...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, eval_mode='argmax', save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "MLPImageStateDiscrete",
        "feature_dim": FEATURE_DIM, "state_dim": STATE_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "chunk_size": CHUNK_SIZE, "action_dim": ACTION_DIM, "n_bins": N_BINS,
        "image_size": IMAGE_SIZE, "n_versions_aug": N_VERSIONS,
        "note": "Run 38 corrigé : 50% v0 (clean, matche eval) + 50% v1..v7 (aug) — fix distribution shift.",
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=20)
    if best_state_dict is not None:
        best_path = run_dir / f"model_best_epoch{best_epoch}.pt"
        torch.save({"state_dict": best_state_dict, "epoch": best_epoch,
                    "success_rate": best_success, "config": model_config}, best_path)
        print(f"  Best model : {best_path} (epoch {best_epoch}, success {best_success:.0%})", flush=True)

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
            "feature_extractor": "ResNet18 (gelé, ImageNet) + augmentation",
            "image_size": IMAGE_SIZE,
            "n_versions_aug": N_VERSIONS,
            "augmentation": data["meta"]["augmentation"],
            "n_bins": N_BINS,
            "n_demos": n_demos,
            "n_train_episodes": n_train_eps, "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch, "best_checkpoint_success": best_success,
            "ce_train_per_epoch": ce_train, "mse_train_per_epoch": mse_train,
            "loss_type": "cross_entropy_per_dim_plus_mse_residual",
            "description": "Run 38 fix : 50% v0 + 50% aug. Cherche à matcher distribution eval clean.",
        },
    )
    save_run(run_data)
    plot_run_losses(run_data, save_dir=str(run_dir))
    generate_run_info(run_data, model, run_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"RÉSUMÉ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Success rate final : {eval_results['success_rate']:.0%} ({int(eval_results['success_rate']*N_EVAL_EPISODES)}/{N_EVAL_EPISODES})", flush=True)
    print(f"  Best checkpoint    : {best_success:.0%} (epoch {best_epoch})", flush=True)
    print(f"  Avg max cube_z     : {eval_results['avg_max_cube_z']:.3f} m", flush=True)
    print(f"  Params entraînés   : {info['total_params']:,}", flush=True)
    print(f"  Temps              : {train_time:.0f}s", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} test={cp['test_loss']:.4f} "
              f"success={cp['success_rate']:.0%} max_z={cp['avg_max_cube_z']:.3f}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
