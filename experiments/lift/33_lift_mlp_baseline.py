"""
Expérience 33 — Baseline MLP sur Robomimic Lift (Phase 3).

Premier baseline imitation sur le dataset Lift PH (200 démos "proficient
human", success 100%). On utilise les observations low-dim (state 19D)
et on prédit un chunk de 20 actions (delta EEF 6D + gripper 1D).

C'est l'équivalent du "run 24" mais pour le bras robotique. Validation
du pipeline complet (loader → MLP → simulation eval) sur une nouvelle
tâche, avant de complexifier (Transformer, image, bins, etc.).

Setup :
- Dataset : lift_ph low_dim (200 démos, ~9.6k frames, 100% success)
- Input  : state 19D (eef_pos + eef_quat + gripper_qpos + object)
- Output : chunk de 20 actions (140 valeurs)
- Loss   : MSE directe (baseline, on sait que limité par PushT mais point de départ)
- Archi  : MLP [19 → 256 → 256 → 140]
- Éval   : Robosuite Lift, success rate sur 50 épisodes, 3 vidéos
"""

import sys
sys.path.insert(0, ".")

import os
import re
import copy
import torch
import torch.nn as nn
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
EXPERIMENT_NAME = "lift/33_lift_mlp_baseline"  # crée results/runs/lift/...
DATASET = "robomimic/lift_ph"

STATE_DIM = 19
ACTION_DIM = 7
CHUNK_SIZE = 20

HIDDEN_DIMS = [256, 256]

LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42

N_EVAL_EPISODES = 50
MAX_STEPS = 200  # Lift demos durent ~50 steps, on prévoit large
CHECKPOINT_EPOCHS = [25, 50, 75, 100]

CACHE_DIR = Path("data_cache")
# ============================================================


class MLPChunkPolicy(nn.Module):
    """MLP qui prédit un chunk d'actions depuis l'état low-dim normalisé."""

    def __init__(self, state_dim, hidden_dims, chunk_size, action_dim,
                 state_mean, state_std, action_mean, action_std):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Stats de normalisation comme buffers (sauvegardés avec state_dict)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

        layers = []
        prev = state_dim
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

    def forward(self, state):
        """state : (batch, state_dim) déjà normalisé. Retourne actions normalisées (batch, chunk, action_dim)."""
        out = self.net(state)
        return out.view(-1, self.chunk_size, self.action_dim)


def build_chunks(states, actions, episodes, chunk_size):
    """Construit les chunks (state_t → 20 actions futures), sans traverser les frontières d'épisode."""
    valid_states, valid_chunks = [], []
    n = len(states)
    i = 0
    while i + chunk_size <= n:
        if episodes[i] == episodes[i + chunk_size - 1]:
            valid_states.append(states[i])
            valid_chunks.append(actions[i:i + chunk_size])
            i += 1
        else:
            i += 1
    X = torch.stack(valid_states)
    Y = torch.stack(valid_chunks)
    print(f"  {len(X)} chunks (chunk_size={chunk_size})", flush=True)
    return X, Y


def split_by_episode(episodes_per_chunk_start, n_demos, train_ratio=0.8, seed=42):
    """Split train/test par épisode (pas par frame) pour éviter leakage.

    episodes_per_chunk_start : (n_chunks,) np.int64 — episode de départ de chaque chunk.
    """
    rng = np.random.default_rng(seed)
    all_eps = np.arange(n_demos)
    rng.shuffle(all_eps)
    n_train_eps = int(train_ratio * n_demos)
    train_eps = set(all_eps[:n_train_eps].tolist())
    test_eps = set(all_eps[n_train_eps:].tolist())

    train_mask = np.array([e in train_eps for e in episodes_per_chunk_start])
    test_mask = np.array([e in test_eps for e in episodes_per_chunk_start])
    train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    test_idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
    return train_idx, test_idx, len(train_eps), len(test_eps)


def train_model(model, X, Y_norm):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n = len(X)
    train_idx, test_idx, n_train_eps, n_test_eps = train_idx_test_idx_global

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
            state_in = model.normalize_state(X[batch])
            pred = model(state_in)  # (B, chunk, action_dim) normalisé
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
            state_in = model.normalize_state(X[test_idx])
            test_pred = model(state_in)
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
                "avg_reward": eval_res["avg_reward"],
                "max_cube_z": eval_res["avg_max_cube_z"],
            })
            print(f"  [CHECKPOINT epoch {epoch+1}] train={avg_train:.4f} test={test_loss:.4f} "
                  f"success={eval_res['success_rate']:.0%} avg_max_z={eval_res['avg_max_cube_z']:.3f}", flush=True)

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
    """Évalue le modèle dans Robosuite Lift."""
    import robosuite as rs

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes)...", flush=True)

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
    table_height = 0.82  # hauteur de la table, pour le seuil de soulèvement

    for ep in range(N_EVAL_EPISODES):
        obs = env.reset()
        # robosuite reset retourne un dict d'obs sans le wrapper gym
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
                    pred_norm = model(state_norm)  # (1, chunk, action_dim)
                    pred_actions = model.denormalize_action(pred_norm).squeeze(0).numpy()  # (chunk, action_dim)
                action_buffer = list(pred_actions)

            action = np.clip(action_buffer.pop(0), -1.0, 1.0)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if "cube_pos" in obs:
                cube_z = obs["cube_pos"][2]
                max_cube_z = max(max_cube_z, cube_z)
            if env._check_success():
                success = True
                # On continue jusqu'au bout pour finir la vidéo
            if done:
                break

        results.append({
            "reward": ep_reward,
            "success": success,
            "max_cube_z": max_cube_z,
        })
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
    label = "mlp_state"
    arch_name = f"MLP {STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → {CHUNK_SIZE*ACTION_DIM} (chunk={CHUNK_SIZE} × action={ACTION_DIM})"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nChargement Lift PH...", flush=True)
    data = load_lift_cached()
    states, actions, episodes = data["states"], data["actions"], data["episodes"]
    stats = data["stats"]
    meta = data["meta"]
    n_demos = meta["n_demos"]

    # Chunks
    print("\nConstruction des chunks...", flush=True)
    X, Y = build_chunks(states, actions, episodes, CHUNK_SIZE)
    # Episodes par chunk (pour split sans leakage)
    chunk_starts = []
    n = len(states)
    i = 0
    while i + CHUNK_SIZE <= n:
        if episodes[i] == episodes[i + CHUNK_SIZE - 1]:
            chunk_starts.append(episodes[i])
            i += 1
        else:
            i += 1
    chunk_starts = np.array(chunk_starts, dtype=np.int64)

    # Split train/test par épisode
    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(chunk_starts, n_demos, train_ratio=0.8, seed=SEED)
    print(f"  Split : {n_train_eps} épisodes train ({len(train_idx)} chunks), {n_test_eps} épisodes test ({len(test_idx)} chunks)", flush=True)

    # Stats de normalisation : calculées sur TRAIN UNIQUEMENT (pas de leakage)
    # train_state_indices = un index global de frame pour chaque chunk train
    # On reprend les states correspondant aux chunks train uniquement
    train_states = X[train_idx]  # (n_train_chunks, 19)
    train_actions = Y[train_idx].reshape(-1, ACTION_DIM)  # (n_train_chunks * 20, 7)
    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0).clamp(min=1e-6)
    action_mean = train_actions.mean(dim=0)
    action_std = train_actions.std(dim=0).clamp(min=1e-6)
    print(f"  state_std (sample) : {state_std[:5].tolist()}", flush=True)
    print(f"  action_std         : {action_std.tolist()}", flush=True)

    # Normalisation des actions (Y) pour le training
    Y_norm = (Y - action_mean) / action_std

    # Modèle
    model = MLPChunkPolicy(
        state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM,
        state_mean=state_mean, state_std=state_std,
        action_mean=action_mean, action_std=action_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,}", flush=True)

    # Exposer le split à train_model (un peu hacky mais évite de refactor)
    global train_idx_test_idx_global
    train_idx_test_idx_global = (train_idx, test_idx, n_train_eps, n_test_eps)

    print("\nEntraînement...", flush=True)
    (train_losses, test_losses, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(model, X, Y_norm)

    # Inférence timing
    dummy_state = torch.randn(1, STATE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            model(model.normalize_state(dummy_state))
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            model(model.normalize_state(dummy_state))
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale avec vidéos...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, save_videos=True)

    # Sauvegarde
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "MLPChunkPolicy",
        "state_dim": STATE_DIM, "hidden_dims": HIDDEN_DIMS,
        "chunk_size": CHUNK_SIZE, "action_dim": ACTION_DIM,
        "note": "Baseline MLP sur state low-dim 19D, prédit chunk 20 actions. Phase 3, ouverture vers bras robotique.",
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
        n_episodes=meta["n_demos"], n_params=info["total_params"], size_mb=info["size_mb"],
        seed=SEED, train_losses=train_losses, test_losses=test_losses,
        train_time_s=train_time, epoch_times=epoch_times,
        flops_total_train=flops_inf * BATCH_SIZE * batches_per_epoch * EPOCHS,
        n_train_samples=n_train, n_test_samples=n_test,
        inference_time_ms=inf_time, flops_inference=flops_inf,
        frames_per_call=CHUNK_SIZE,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_max_cube_z"],  # on réutilise le champ avg_coverage pour la hauteur max
        avg_reward=eval_results["avg_reward"],
        extra={
            "task": "Robomimic Lift (cube + Panda 7-DoF)",
            "state_keys": list(STATE_KEYS),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "n_demos": meta["n_demos"],
            "n_train_episodes": n_train_eps,
            "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch,
            "best_checkpoint_success": best_success,
            "loss_type": "mse_on_normalized_actions",
            "description": "Phase 3 — baseline MLP imitation sur Robomimic Lift. Premier transfert depuis PushT.",
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
        print(f"    epoch {cp['epoch']:3d} : train={cp['train_loss']:.4f} test={cp['test_loss']:.4f} "
              f"success={cp['success_rate']:.0%} max_z={cp['max_cube_z']:.3f}", flush=True)
    print(f"\n  Dossier : {run_dir}", flush=True)


if __name__ == "__main__":
    main()
