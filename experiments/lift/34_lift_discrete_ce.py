"""
Expérience 34 — Lift CE + résidu sur bins discrets (applique le winning formula de run 31).

Variante de 33_lift_mlp_baseline :
- Au lieu d'une régression MSE sur actions continues,
  on discrétise chaque dim d'action [-1, +1] en N=21 bins uniformes,
  et on prédit bin (classification) + résidu (offset fin dans le bin).
- C'est exactement la formule de run 31 (PushT, 44.9% argmax) adaptée au 7D.

Différences vs run 33 :
- Sortie : (chunk, 7, 21) logits + (chunk, 7, 21) résidus, au lieu de (chunk, 7) continu
- Loss : CE sur bin + MSE sur résidu (somme), au lieu de MSE continu
- Inférence : argmax bin → add résidu → clip à [-1, +1]

Objectif : vérifier que la leçon "CE bat MSE sur multimodal" transfère du 2D PushT
au 7D Lift, et casser le mode collapse identifié en run 33 (0% success).
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
from src.lift_data import load_lift_cached, build_state_vector, STATE_KEYS
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir
)
from src.visualize import plot_run_losses
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "lift/34_lift_discrete_ce"
DATASET = "robomimic/lift_ph"

STATE_DIM = 19
ACTION_DIM = 7
CHUNK_SIZE = 20

HIDDEN_DIMS = [256, 256]
N_BINS = 21               # 21 bins uniformes dans [-1, +1], pas = 0.1
RESIDUAL_WEIGHT = 1.0     # pondération MSE résidu vs CE

LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

N_EVAL_EPISODES = 50
MAX_STEPS = 200
CHECKPOINT_EPOCHS = [25, 50, 75, 100]

CACHE_DIR = Path("data_cache")
# ============================================================


def build_bin_centers(n_bins):
    """Centres des bins uniformes dans [-1, +1]."""
    edges = torch.linspace(-1.0, 1.0, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2  # (n_bins,)


def assign_bins(actions, n_bins):
    """Index du bin pour chaque action (dans [-1, +1])."""
    bin_width = 2.0 / n_bins
    idx = ((actions + 1.0) / bin_width).long()
    return idx.clamp(0, n_bins - 1)


class MLPDiscretePolicy(nn.Module):
    """MLP + sortie discrète par dim d'action (CE+résidu, style run 31 PushT)."""

    def __init__(self, state_dim, hidden_dims, chunk_size, action_dim, n_bins,
                 state_mean, state_std):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_bins = n_bins

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("bin_centers", build_bin_centers(n_bins))

        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.body = nn.Sequential(*layers)
        self.bin_head = nn.Linear(prev, chunk_size * action_dim * n_bins)
        self.residual_head = nn.Linear(prev, chunk_size * action_dim * n_bins)

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def forward(self, state):
        """Retourne (bin_logits, residuals) de shape (B, chunk, action_dim, n_bins) chacun."""
        h = self.body(state)
        bin_logits = self.bin_head(h).view(-1, self.chunk_size, self.action_dim, self.n_bins)
        residuals = self.residual_head(h).view(-1, self.chunk_size, self.action_dim, self.n_bins)
        return bin_logits, residuals

    def compute_loss(self, bin_logits, residuals, target_actions):
        """target_actions: (B, chunk, action_dim) dans [-1, +1]."""
        B, chunk, A, N = bin_logits.shape
        target_bins = assign_bins(target_actions, self.n_bins)  # (B, chunk, action_dim)

        # CE per (B, chunk, action_dim) entry
        ce_loss = F.cross_entropy(
            bin_logits.reshape(-1, N),
            target_bins.reshape(-1),
        )

        # Résidu cible = action - centre du bin choisi
        target_residuals = target_actions - self.bin_centers[target_bins]
        # Sélectionne le résidu prédit au bon bin
        idx = target_bins.unsqueeze(-1)  # (B, chunk, action_dim, 1)
        sel_residual = torch.gather(residuals, dim=-1, index=idx).squeeze(-1)
        mse_loss = F.mse_loss(sel_residual, target_residuals)
        return ce_loss + RESIDUAL_WEIGHT * mse_loss, ce_loss.item(), mse_loss.item()

    def predict(self, bin_logits, residuals, mode='argmax'):
        """Retourne actions (B, chunk, action_dim) dans [-1, +1]."""
        if mode == 'argmax':
            bin_ids = bin_logits.argmax(dim=-1)  # (B, chunk, action_dim)
        else:
            probs = F.softmax(bin_logits, dim=-1)
            flat_probs = probs.reshape(-1, self.n_bins)
            bin_ids = torch.multinomial(flat_probs, 1).reshape(*probs.shape[:-1])
        centers = self.bin_centers[bin_ids]  # (B, chunk, action_dim)
        idx = bin_ids.unsqueeze(-1)
        sel_residual = torch.gather(residuals, dim=-1, index=idx).squeeze(-1)
        return (centers + sel_residual).clamp(-1.0, 1.0)


def build_chunks(states, actions, episodes, chunk_size):
    """Construit les chunks (state_t → 20 actions futures), sans traverser les frontières."""
    valid_states, valid_chunks, chunk_episodes = [], [], []
    n = len(states)
    i = 0
    while i + chunk_size <= n:
        if episodes[i] == episodes[i + chunk_size - 1]:
            valid_states.append(states[i])
            valid_chunks.append(actions[i:i + chunk_size])
            chunk_episodes.append(int(episodes[i]))
            i += 1
        else:
            i += 1
    X = torch.stack(valid_states)
    Y = torch.stack(valid_chunks)
    chunk_eps = np.array(chunk_episodes, dtype=np.int64)
    print(f"  {len(X)} chunks (chunk_size={chunk_size})", flush=True)
    return X, Y, chunk_eps


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


def train_model(model, X, Y, train_idx, test_idx):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, test_losses = [], []
    ce_train, mse_train = [], []
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
        epoch_loss, epoch_ce, epoch_mse, n_batches = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            state_in = model.normalize_state(X[batch])
            bin_logits, residuals = model(state_in)
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
            state_in = model.normalize_state(X[test_idx])
            bin_logits, residuals = model(state_in)
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
            eval_res, _ = eval_in_simulation(model, eval_mode='argmax', save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "test_loss": test_loss,
                "ce_loss": avg_ce,
                "mse_loss": avg_mse,
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

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes, mode={eval_mode})...", flush=True)

    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=256, camera_widths=256,
        control_freq=20, reward_shaping=False,
    )

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
                state_np = build_state_vector(obs).astype(np.float32)
                state_tensor = torch.tensor(state_np).unsqueeze(0)
                with torch.no_grad():
                    state_norm = model.normalize_state(state_tensor)
                    bin_logits, residuals = model(state_norm)
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
    label = "mlp_ce_residual"
    arch_name = f"MLP {STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → CE+résidu (N={N_BINS} bins × {ACTION_DIM} dims × chunk={CHUNK_SIZE})"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nChargement Lift PH...", flush=True)
    data = load_lift_cached()
    states, actions, episodes = data["states"], data["actions"], data["episodes"]
    n_demos = data["meta"]["n_demos"]

    print("\nConstruction des chunks...", flush=True)
    X, Y, chunk_eps = build_chunks(states, actions, episodes, CHUNK_SIZE)
    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(chunk_eps, n_demos, train_ratio=0.8, seed=SEED)
    print(f"  Split : {n_train_eps} démos train ({len(train_idx)} chunks), {n_test_eps} démos test ({len(test_idx)} chunks)", flush=True)

    # Stats normalisation sur train uniquement
    train_states = X[train_idx]
    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0).clamp(min=1e-6)

    model = MLPDiscretePolicy(
        state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM, n_bins=N_BINS,
        state_mean=state_mean, state_std=state_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)
    print(f"  N_BINS={N_BINS} → output_dim/timestep = {ACTION_DIM*N_BINS} logits + {ACTION_DIM*N_BINS} résidus = {2*ACTION_DIM*N_BINS}", flush=True)

    print("\nEntraînement...", flush=True)
    (train_losses, test_losses, ce_train, mse_train, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(model, X, Y, train_idx, test_idx)

    # Inference timing
    dummy_state = torch.randn(1, STATE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            state_norm = model.normalize_state(dummy_state)
            bl, rs_ = model(state_norm)
            model.predict(bl, rs_, mode='argmax')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            state_norm = model.normalize_state(dummy_state)
            bl, rs_ = model(state_norm)
            model.predict(bl, rs_, mode='argmax')
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale (argmax + vidéos)...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, eval_mode='argmax', save_videos=True)

    # Sauvegarde
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "MLPDiscretePolicy",
        "state_dim": STATE_DIM, "hidden_dims": HIDDEN_DIMS,
        "chunk_size": CHUNK_SIZE, "action_dim": ACTION_DIM,
        "n_bins": N_BINS, "residual_weight": RESIDUAL_WEIGHT,
        "note": "Run 31 PushT formula applied to Lift: CE per dim + residual per bin.",
    }
    save_model(model, run_dir, model_config)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=20)

    if best_state_dict is not None:
        best_path = run_dir / f"model_best_epoch{best_epoch}.pt"
        torch.save({
            "state_dict": best_state_dict,
            "epoch": best_epoch,
            "success_rate": best_success,
            "config": model_config,
        }, best_path)
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
            "state_keys": list(STATE_KEYS),
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "n_bins": N_BINS,
            "n_demos": n_demos,
            "n_train_episodes": n_train_eps, "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch, "best_checkpoint_success": best_success,
            "ce_train_per_epoch": ce_train, "mse_train_per_epoch": mse_train,
            "loss_type": "cross_entropy_per_dim_plus_mse_residual",
            "description": "Lift CE + résidu (formule run 31 PushT appliquée au 7D Lift).",
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
    print(f"  Params             : {info['total_params']:,}", flush=True)
    print(f"  Temps              : {train_time:.0f}s", flush=True)
    print(f"\n  Checkpoint evals :", flush=True)
    for cp in checkpoint_evals:
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} "
              f"(ce={cp['ce_loss']:.4f} mse={cp['mse_loss']:.4f}) test={cp['test_loss']:.4f} "
              f"success={cp['success_rate']:.0%} max_z={cp['avg_max_cube_z']:.3f}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
