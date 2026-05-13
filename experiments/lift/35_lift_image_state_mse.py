"""
Expérience 35 — Lift MLP sur features image ResNet18 + state low-dim (équivalent run 24 PushT).

Mêmes idées que run 24 PushT mais sur Lift :
- Pré-calcul des features ResNet18 (gelé) sur les images rendues à partir des states du dataset
- MLP qui prend [features 512D + state 19D = 531D] → chunk 20 actions
- Loss MSE sur actions normalisées (baseline)

But : vérifier que l'apport image résout l'overfit dur de run 33 (state-only MSE, 0% success).

Hypothèse : avec ResNet18 frozen comme encodeur visuel, le modèle a accès
à une information plus riche et redondante → meilleure généralisation
malgré le petit dataset (200 démos).
"""

import sys
sys.path.insert(0, ".")

import copy
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.lift_image_features import extract_lift_image_features
from src.lift_data import build_state_vector, STATE_KEYS
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "lift/35_lift_image_state_mse"
DATASET = "robomimic/lift_ph"

FEATURE_DIM = 512
STATE_DIM = 19
ACTION_DIM = 7
CHUNK_SIZE = 20
IMAGE_SIZE = 96

HIDDEN_DIMS = [256, 256]

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


class MLPImageStatePolicy(nn.Module):
    """MLP qui prend (features ResNet image + state low-dim) → chunk d'actions."""

    def __init__(self, feature_dim, state_dim, hidden_dims, chunk_size, action_dim,
                 state_mean, state_std, action_mean, action_std):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

        in_dim = feature_dim + state_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def forward(self, features, state):
        """features: (B, 512), state: (B, 19) déjà normalisé."""
        x = torch.cat([features, state], dim=-1)
        out = self.net(x)
        return out.view(-1, self.chunk_size, self.action_dim)


def build_chunks(features, states, actions, episodes, chunk_size):
    """Construit les chunks (features_t, state_t → 20 actions futures)."""
    valid_feats, valid_states, valid_chunks, chunk_eps = [], [], [], []
    n = len(features)
    i = 0
    while i + chunk_size <= n:
        if episodes[i] == episodes[i + chunk_size - 1]:
            valid_feats.append(features[i])
            valid_states.append(states[i])
            valid_chunks.append(actions[i:i + chunk_size])
            chunk_eps.append(int(episodes[i]))
            i += 1
        else:
            i += 1
    XF = torch.stack(valid_feats)
    XS = torch.stack(valid_states)
    Y = torch.stack(valid_chunks)
    chunk_eps = np.array(chunk_eps, dtype=np.int64)
    print(f"  {len(XF)} chunks (chunk_size={chunk_size})", flush=True)
    return XF, XS, Y, chunk_eps


def split_by_episode(chunk_episodes, n_demos, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    all_eps = np.arange(n_demos)
    rng.shuffle(all_eps)
    n_train_eps = int(train_ratio * n_demos)
    train_eps = set(all_eps[:n_train_eps].tolist())
    test_eps = set(all_eps[n_train_eps:].tolist())
    train_mask = np.array([e in train_eps for e in chunk_episodes])
    test_mask = np.array([e in test_eps for e in chunk_episodes])
    train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    test_idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
    return train_idx, test_idx, len(train_eps), len(test_eps)


def train_model(model, XF, XS, Y_norm, train_idx, test_idx):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    epoch_times = []
    checkpoint_evals = []
    best_success = -1.0
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
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            state_in = model.normalize_state(XS[batch])
            pred = model(XF[batch], state_in)
            loss = criterion(pred, Y_norm[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            state_in = model.normalize_state(XS[test_idx])
            test_pred = model(XF[test_idx], state_in)
            test_loss = criterion(test_pred, Y_norm[test_idx]).item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train)
        test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train: {avg_train:.6f} — test: {test_loss:.6f} — {elapsed:.0f}s", flush=True)

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            print(f"\n  [CHECKPOINT] Éval à epoch {epoch+1}...", flush=True)
            eval_res, _ = eval_in_simulation(model, save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "test_loss": test_loss,
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
    return (train_losses, test_losses, epoch_times, total_time,
            len(train_idx), len(test_idx), checkpoint_evals,
            best_state_dict, best_success, best_epoch)


def eval_in_simulation(model, save_videos=True):
    """Évalue avec sim Robosuite Lift : à chaque step, render image → ResNet → MLP."""
    import robosuite as rs
    from torchvision.models import resnet18, ResNet18_Weights
    import torch.nn.functional as F

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes)...", flush=True)

    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )

    # ResNet18 gelé pour l'inférence live
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
                # Render image pour features
                img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
                img_t = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_t = (img_t - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    features = backbone(img_t).flatten(1)  # (1, 512)
                # State
                state_np = build_state_vector(obs).astype(np.float32)
                state_tensor = torch.tensor(state_np).unsqueeze(0)
                state_norm = model.normalize_state(state_tensor)
                # Prédire chunk
                with torch.no_grad():
                    pred_norm = model(features, state_norm)
                    pred_actions = model.denormalize_action(pred_norm).squeeze(0).numpy()
                action_buffer = list(pred_actions)

            action = np.clip(action_buffer.pop(0), -1.0, 1.0)
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
    label = "mlp_feat_state_mse"
    arch_name = f"ResNet18(gelé) + MLP {FEATURE_DIM+STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → {CHUNK_SIZE*ACTION_DIM}"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nChargement features ResNet + states...", flush=True)
    data = extract_lift_image_features(image_size=IMAGE_SIZE)
    features, states, actions, episodes = data["features"], data["states"], data["actions"], data["episodes"]
    n_demos = data["meta"]["n_demos"]

    print("\nConstruction des chunks...", flush=True)
    XF, XS, Y, chunk_eps = build_chunks(features, states, actions, episodes, CHUNK_SIZE)
    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(chunk_eps, n_demos, train_ratio=0.8, seed=SEED)
    print(f"  Split : {n_train_eps} démos train ({len(train_idx)} chunks), {n_test_eps} démos test ({len(test_idx)} chunks)", flush=True)

    # Stats normalisation sur train uniquement (states + actions)
    state_mean = XS[train_idx].mean(dim=0)
    state_std = XS[train_idx].std(dim=0).clamp(min=1e-6)
    train_actions_flat = Y[train_idx].reshape(-1, ACTION_DIM)
    action_mean = train_actions_flat.mean(dim=0)
    action_std = train_actions_flat.std(dim=0).clamp(min=1e-6)

    Y_norm = (Y - action_mean) / action_std

    model = MLPImageStatePolicy(
        feature_dim=FEATURE_DIM, state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM,
        state_mean=state_mean, state_std=state_std,
        action_mean=action_mean, action_std=action_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,} (ResNet18 = 11.2M, gelés)", flush=True)

    print("\nEntraînement...", flush=True)
    (train_losses, test_losses, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(model, XF, XS, Y_norm, train_idx, test_idx)

    # Inference timing
    dummy_feat = torch.randn(1, FEATURE_DIM)
    dummy_state = torch.randn(1, STATE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            model(dummy_feat, model.normalize_state(dummy_state))
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            model(dummy_feat, model.normalize_state(dummy_state))
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale avec vidéos...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "MLPImageStatePolicy",
        "feature_dim": FEATURE_DIM, "state_dim": STATE_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "chunk_size": CHUNK_SIZE, "action_dim": ACTION_DIM,
        "image_size": IMAGE_SIZE,
        "note": "Équivalent run 24 PushT : features ResNet18 (gelé) + state → MLP → chunk actions. Loss MSE.",
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
            "feature_extractor": "ResNet18 (gelé, ImageNet)",
            "image_size": IMAGE_SIZE,
            "state_dim": STATE_DIM, "feature_dim": FEATURE_DIM, "action_dim": ACTION_DIM,
            "n_demos": n_demos,
            "n_train_episodes": n_train_eps, "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch, "best_checkpoint_success": best_success,
            "loss_type": "mse_on_normalized_actions",
            "description": "Lift baseline image + state (équivalent run 24 PushT). MLP sur features ResNet pré-calculées.",
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
