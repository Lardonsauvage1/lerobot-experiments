"""
Expérience 13 — Évaluer le Diffusion Policy pré-entraîné de LeRobot sur PushT.

C'est le modèle de référence "état de l'art" pour PushT.
On ne l'entraîne pas — on le charge et on le teste dans la simulation
pour voir quel niveau de performance est atteignable.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from src.tracker import (
    build_run_data, save_run, save_model, generate_run_info,
    save_eval_videos, get_run_dir, measure_inference_time
)
from src.benchmark import model_info

# ============================================================
EXPERIMENT_NAME = "13_diffusion_pretrained"
DATASET = "lerobot/pusht"
IMAGE_SIZE = 96  # Diffusion Policy utilise 96x96
SEED = 42
N_EVAL_EPISODES = 10
MAX_STEPS = 300
# ============================================================


def load_diffusion_policy():
    """Charge le Diffusion Policy pré-entraîné depuis le Hub."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    print("  Chargement du modèle depuis lerobot/diffusion_pusht...")
    # Forcer CPU avant le chargement
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained("lerobot/diffusion_pusht")
    config.device = "cpu"

    policy = DiffusionPolicy(config)
    policy = policy.to("cpu")

    # Charger les poids (strict=False car ancien format avec clés de normalisation)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    weights_path = hf_hub_download("lerobot/diffusion_pusht", "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in policy.parameters()) / (1024 * 1024)
    print(f"  Paramètres : {total_params:,}")
    print(f"  Taille     : {size_mb:.1f} Mo")

    return policy, total_params, size_mb


def eval_in_simulation(policy):
    """Teste le Diffusion Policy dans la simulation PushT."""
    import gymnasium as gym
    import gym_pusht  # noqa

    print(f"\n  Évaluation simulation ({N_EVAL_EPISODES} épisodes)...")

    # Le Diffusion Policy attend des observations normalisées
    # On doit utiliser les stats du dataset pour normaliser
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/pusht")
    stats = dataset.meta.stats

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    results = []
    all_frames = []
    inference_times = []

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep * 42)
        frames = []
        ep_reward = 0
        max_coverage = 0

        policy.reset()

        for step in range(MAX_STEPS):
            if ep < 3:
                frames.append(env.render())

            # Préparer l'observation dans le format attendu par le policy
            # Image : (96, 96, 3) uint8 → (1, 3, 96, 96) float32 normalisé
            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            # Agent pos : (2,) float64 → (1, 2) float32
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)

            # Normaliser avec les stats du dataset
            img_mean = torch.tensor(stats["observation.image"]["mean"], dtype=torch.float32)
            img_std = torch.tensor(stats["observation.image"]["std"], dtype=torch.float32)
            img_std = torch.where(img_std > 1e-6, img_std, torch.ones_like(img_std))

            state_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32)
            state_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32)
            state_std = torch.where(state_std > 1e-6, state_std, torch.ones_like(state_std))

            img_norm = (img - img_mean) / img_std
            pos_norm = (agent_pos - state_mean) / state_std

            observation = {
                "observation.image": img_norm.unsqueeze(0),
                "observation.state": pos_norm.unsqueeze(0),
            }

            # Inférence
            t0 = time.time()
            with torch.inference_mode():
                action_out = policy.select_action(observation)
            inf_time = (time.time() - t0) * 1000
            inference_times.append(inf_time)

            # Dénormaliser l'action
            action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
            action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
            # select_action retourne un tensor (batch, 2) ou un dict
            if isinstance(action_out, dict):
                action_tensor = action_out["action"]
            else:
                action_tensor = action_out
            action = (action_tensor.squeeze(0) * action_std + action_mean).numpy()

            action = np.clip(action, 0, 512).astype(np.float32)
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
    avg_inf_time = np.mean(inference_times)

    print(f"\n    Success rate    : {n_success}/{N_EVAL_EPISODES} ({n_success/N_EVAL_EPISODES:.0%})")
    print(f"    Coverage moyen  : {avg_coverage:.1%}")
    print(f"    Reward moyen    : {avg_reward:.2f}")
    print(f"    Inférence moyen : {avg_inf_time:.1f}ms")

    return {
        "success_rate": n_success / N_EVAL_EPISODES,
        "avg_reward": float(avg_reward),
        "avg_coverage": float(avg_coverage),
        "avg_inference_ms": float(avg_inf_time),
    }, all_frames


def main():
    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {EXPERIMENT_NAME}")
    print(f"Diffusion Policy pré-entraîné (lerobot/diffusion_pusht)")
    print(f"{'='*60}")

    # 1. Charger
    policy, total_params, size_mb = load_diffusion_policy()

    # 2. Évaluer
    eval_results, all_frames = eval_in_simulation(policy)

    # 3. Sauvegarder
    run_dir = get_run_dir(EXPERIMENT_NAME, "diffusion_pusht")

    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=10)

    run_data = build_run_data(
        experiment=EXPERIMENT_NAME,
        architecture="Diffusion Policy (U-Net 1D, pré-entraîné)",
        dataset=DATASET,
        n_episodes="206 (pré-entraîné par LeRobot)",
        n_params=total_params,
        size_mb=size_mb,
        seed=SEED,
        train_losses=[],
        test_losses=[],
        train_time_s=0,
        epoch_times=[],
        flops_total_train=0,
        n_train_samples=0,
        n_test_samples=0,
        inference_time_ms=eval_results["avg_inference_ms"],
        flops_inference=0,
        frames_per_call=1,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_coverage"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "pretrained": "lerobot/diffusion_pusht",
            "description": "Diffusion Policy pré-entraîné par LeRobot — référence état de l'art pour PushT",
        },
    )
    save_run(run_data)
    generate_run_info(run_data, policy, run_dir)

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ — Diffusion Policy pré-entraîné")
    print(f"{'='*60}")
    print(f"  Success rate  : {eval_results['success_rate']:.0%}")
    print(f"  Coverage      : {eval_results['avg_coverage']:.1%}")
    print(f"  Paramètres    : {total_params:,}")
    print(f"  Taille        : {size_mb:.1f} Mo")
    print(f"  Inférence     : {eval_results['avg_inference_ms']:.1f}ms")
    print(f"  Dossier       : {run_dir}")


if __name__ == "__main__":
    main()
