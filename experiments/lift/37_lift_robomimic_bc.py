"""
Expérience 37 — Lift : BC style Robomimic (référence externe).

Clone de la recette BC du papier Robomimic (Mandlekar et al. 2021) :
- chunk = 1 (re-prédire chaque step → vraie receding horizon)
- GMM head K=5 (mixture of 5 gaussiennes diagonales)
- MLP [1024, 1024] + LayerNorm + ReLU
- Loss = NLL du mélange gaussien
- LR 1e-4, batch 100, Adam, 500 epochs
- Input : features ResNet18 + state low-dim

Ce setup atteint ~95% success sur lift_ph dans la littérature. C'est la
référence externe pour comprendre l'écart avec nos approches précédentes
(run 35 MSE → 28%, run 36 CE → 34%).
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
EXPERIMENT_NAME = "lift/37_lift_robomimic_bc"
DATASET = "robomimic/lift_ph"

FEATURE_DIM = 512
STATE_DIM = 19
ACTION_DIM = 7
IMAGE_SIZE = 96

HIDDEN_DIMS = [1024, 1024]   # ↑ vs 256 pour matcher Robomimic
K_MIXTURES = 5               # GMM K=5 (défaut Robomimic)
LOG_SIGMA_MIN = -5.0
LOG_SIGMA_MAX = 2.0

LEARNING_RATE = 1e-4         # ↓ vs 1e-3 pour stabilité avec gros MLP
WARMUP_EPOCHS = 20
GRAD_CLIP = 1.0
EPOCHS = 500                 # ↑ Robomimic entraîne longtemps
BATCH_SIZE = 100             # Robomimic default
SEED = 42

N_EVAL_EPISODES = 50
MAX_STEPS = 200
CHECKPOINT_EPOCHS = [100, 200, 300, 400, 500]
# ============================================================


class BCRobomimic(nn.Module):
    """Behavioral Cloning style Robomimic : MLP + GMM head.

    Forward : (features, state) → (pi_logits (B,K), mu (B,K,A), log_sigma (B,K,A))
    """

    def __init__(self, feature_dim, state_dim, hidden_dims, action_dim, k_mixtures,
                 state_mean, state_std):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k_mixtures

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)

        in_dim = feature_dim + state_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))      # Robomimic utilise LayerNorm
            layers.append(nn.ReLU())
            prev = h
        self.body = nn.Sequential(*layers)
        # GMM head : K * (1 weight + action_dim mu + action_dim log_sigma)
        self.gmm_head = nn.Linear(prev, k_mixtures * (1 + 2 * action_dim))

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def forward(self, features, state):
        x = torch.cat([features, state], dim=-1)
        h = self.body(x)
        out = self.gmm_head(h).view(-1, self.k, 1 + 2 * self.action_dim)
        pi_logits = out[..., 0]                                  # (B, K)
        mu = out[..., 1:1 + self.action_dim]                     # (B, K, A)
        log_sigma = out[..., 1 + self.action_dim:]               # (B, K, A)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return pi_logits, mu, log_sigma

    def gmm_nll(self, pi_logits, mu, log_sigma, target):
        """NLL d'un mélange de K gaussiennes diagonales.

        target: (B, action_dim)
        """
        target_exp = target.unsqueeze(1)  # (B, 1, A)
        diff = target_exp - mu             # (B, K, A)
        log_prob_per_dim = (-0.5 * (diff * torch.exp(-log_sigma)) ** 2
                            - log_sigma
                            - 0.5 * math.log(2 * math.pi))
        log_prob = log_prob_per_dim.sum(dim=-1)  # (B, K)
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()

    def sample(self, pi_logits, mu, log_sigma, mode='sample'):
        """Retourne une action (B, action_dim) échantillonnée du mélange."""
        pi = F.softmax(pi_logits, dim=-1)
        if mode == 'argmax':
            idx = pi.argmax(dim=-1)  # (B,)
        else:
            idx = torch.multinomial(pi, 1).squeeze(-1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, mu.shape[-1])  # (B, 1, A)
        sel_mu = torch.gather(mu, dim=1, index=idx_exp).squeeze(1)
        sel_log_sigma = torch.gather(log_sigma, dim=1, index=idx_exp).squeeze(1)
        if mode == 'mean':
            return sel_mu
        return sel_mu + torch.randn_like(sel_mu) * torch.exp(sel_log_sigma)


def split_by_episode(episodes, n_demos, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    all_eps = np.arange(n_demos)
    rng.shuffle(all_eps)
    n_train_eps = int(train_ratio * n_demos)
    train_eps = set(all_eps[:n_train_eps].tolist())
    test_eps = set(all_eps[n_train_eps:].tolist())
    train_mask = np.array([e in train_eps for e in episodes])
    test_mask = np.array([e in test_eps for e in episodes])
    return (torch.tensor(np.where(train_mask)[0], dtype=torch.long),
            torch.tensor(np.where(test_mask)[0], dtype=torch.long),
            len(train_eps), len(test_eps))


def train_model(model, XF, XS, Y, train_idx, test_idx):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, test_losses = [], []
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
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), BATCH_SIZE):
            batch = train_idx[perm[i:i + BATCH_SIZE]]
            state_in = model.normalize_state(XS[batch])
            pi_logits, mu, log_sigma = model(XF[batch], state_in)
            loss = model.gmm_nll(pi_logits, mu, log_sigma, Y[batch])
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
            pi_logits, mu, log_sigma = model(XF[test_idx], state_in)
            test_loss = model.gmm_nll(pi_logits, mu, log_sigma, Y[test_idx]).item()

        ep_time = time.time() - start_ep
        train_losses.append(avg_train); test_losses.append(test_loss)
        epoch_times.append(ep_time)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start_total
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train_nll: {avg_train:.4f} test_nll: {test_loss:.4f} — {elapsed:.0f}s", flush=True)

        if (epoch + 1) in CHECKPOINT_EPOCHS:
            print(f"\n  [CHECKPOINT] Éval à epoch {epoch+1}...", flush=True)
            eval_res, _ = eval_in_simulation(model, eval_mode='sample', save_videos=False)
            checkpoint_evals.append({
                "epoch": epoch + 1, "train_loss": avg_train, "test_loss": test_loss,
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


def eval_in_simulation(model, eval_mode='sample', save_videos=True):
    """Eval avec re-prédiction à chaque step (chunk=1)."""
    import robosuite as rs
    from torchvision.models import resnet18, ResNet18_Weights

    print(f"  Évaluation ({N_EVAL_EPISODES} épisodes, mode={eval_mode}, re-prédiction chaque step)...", flush=True)

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

        for step in range(MAX_STEPS):
            if save_videos and ep < 3:
                frame = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
                frames.append(frame)

            # Prédire 1 action depuis état courant
            img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
            img_t = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_t = (img_t - imagenet_mean) / imagenet_std
            with torch.no_grad():
                features = backbone(img_t).flatten(1)
            state_np = build_state_vector(obs).astype(np.float32)
            state_tensor = torch.tensor(state_np).unsqueeze(0)
            state_norm = model.normalize_state(state_tensor)
            with torch.no_grad():
                pi_logits, mu, log_sigma = model(features, state_norm)
                action = model.sample(pi_logits, mu, log_sigma, mode=eval_mode).squeeze(0).numpy()
            action = np.clip(action, -1.0, 1.0)

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
    label = "robomimic_bc_gmm"
    arch_name = f"ResNet18(gelé) + MLP {FEATURE_DIM+STATE_DIM} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → GMM(K={K_MIXTURES})"

    print(f"\n{'='*60}", flush=True)
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}", flush=True)
    print(f"{'='*60}", flush=True)
    torch.manual_seed(SEED)

    print("\nChargement features ResNet + states...", flush=True)
    data = extract_lift_image_features(image_size=IMAGE_SIZE)
    features, states, actions, episodes = data["features"], data["states"], data["actions"], data["episodes"]
    n_demos = data["meta"]["n_demos"]
    print(f"  Frames : {len(features)} (chunk=1, pas de chunking)", flush=True)

    train_idx, test_idx, n_train_eps, n_test_eps = split_by_episode(episodes, n_demos, train_ratio=0.8, seed=SEED)
    print(f"  Split : {n_train_eps} démos train ({len(train_idx)} frames), {n_test_eps} démos test ({len(test_idx)} frames)", flush=True)

    state_mean = states[train_idx].mean(dim=0)
    state_std = states[train_idx].std(dim=0).clamp(min=1e-6)

    model = BCRobomimic(
        feature_dim=FEATURE_DIM, state_dim=STATE_DIM, hidden_dims=HIDDEN_DIMS,
        action_dim=ACTION_DIM, k_mixtures=K_MIXTURES,
        state_mean=state_mean, state_std=state_std,
    )
    info = model_info(model, name=arch_name)
    print(f"  Paramètres entraînés : {info['total_params']:,} (ResNet18 = 11.2M, gelés)", flush=True)

    print("\nEntraînement (GMM NLL, 500 epochs, LR 1e-4)...", flush=True)
    (train_losses, test_losses, epoch_times, train_time,
     n_train, n_test, checkpoint_evals,
     best_state_dict, best_success, best_epoch) = train_model(model, features, states, actions, train_idx, test_idx)

    dummy_feat = torch.randn(1, FEATURE_DIM)
    dummy_state = torch.randn(1, STATE_DIM)
    model.eval()
    for _ in range(20):
        with torch.no_grad():
            pl, m, ls = model(dummy_feat, model.normalize_state(dummy_state))
            model.sample(pl, m, ls, mode='sample')
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            pl, m, ls = model(dummy_feat, model.normalize_state(dummy_state))
            model.sample(pl, m, ls, mode='sample')
    inf_time = (time.time() - t0) / 200 * 1000

    print("\nÉval finale avec vidéos...", flush=True)
    eval_results, all_frames = eval_in_simulation(model, eval_mode='sample', save_videos=True)

    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    model_config = {
        "class": "BCRobomimic",
        "feature_dim": FEATURE_DIM, "state_dim": STATE_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "action_dim": ACTION_DIM, "k_mixtures": K_MIXTURES,
        "image_size": IMAGE_SIZE,
        "note": "Clone BC Robomimic : MLP[1024,1024]+LN+ReLU + GMM K=5, chunk=1, NLL.",
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
        frames_per_call=1,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_max_cube_z"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "task": "Robomimic Lift (Panda 7-DoF + cube)",
            "feature_extractor": "ResNet18 (gelé, ImageNet)",
            "image_size": IMAGE_SIZE,
            "k_mixtures": K_MIXTURES,
            "n_demos": n_demos,
            "n_train_episodes": n_train_eps, "n_test_episodes": n_test_eps,
            "checkpoint_evals": checkpoint_evals,
            "best_checkpoint_epoch": best_epoch, "best_checkpoint_success": best_success,
            "loss_type": "gmm_nll",
            "description": "Clone de la recette BC du papier Robomimic (chunk=1, GMM, MLP[1024,1024]+LN).",
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
