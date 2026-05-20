"""
Eval run 46 — Diffusion Policy entraînée via lerobot-train sur notre Lift PH.

Charge un checkpoint LeRobot DiffusionPolicy, le fait tourner dans notre
env Robosuite Lift (pas l'env LeRobot car Lift non supporté en v0.5),
mesure success rate sur 200 episodes.

Compatible avec :
- Checkpoint final (results/runs/lift/46_diffusion_official/checkpoints/last/pretrained_model)
- Checkpoints intermédiaires (results/runs/lift/46_diffusion_official/checkpoints/<step>/pretrained_model)
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import time
import torch
import numpy as np
import imageio
from pathlib import Path

from src.lift_data import build_state_vector
from src.tracker import build_run_data, save_run, save_eval_videos, get_run_dir

# ============================================================
DEFAULT_CHECKPOINT = "results/runs/lift/46_diffusion_official/checkpoints/last/pretrained_model"
N_EVAL_EPISODES = 200
IMAGE_SIZE = 96
MAX_STEPS = 200
N_VIDEOS = 3
NUM_INFERENCE_STEPS = 10  # DDIM standard literature ; défaut buggy = 100 (= num_train_timesteps)
# ============================================================


def load_policy(checkpoint_path, device):
    """Charge la DiffusionPolicy + preprocessor + postprocessor depuis un checkpoint LeRobot.

    LeRobot v0.5 a séparé les normalizers en pipelines externes. `policy.select_action()`
    seul ne normalise PAS observation ni dé-normalise action. Il faut appliquer manuellement
    preprocessor(obs) → policy.select_action(...) → postprocessor(action).
    """
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    print(f"Loading Diffusion Policy from {checkpoint_path}...", flush=True)
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device).eval()
    policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy : {n_params:,} params, device={device}", flush=True)
    print(f"  Config : horizon={policy.config.horizon}, n_action_steps={policy.config.n_action_steps}, "
          f"num_inference_steps={policy.diffusion.num_inference_steps}", flush=True)
    print(f"  Preprocessor  : {type(preprocessor).__name__}", flush=True)
    print(f"  Postprocessor : {type(postprocessor).__name__}", flush=True)
    return policy, preprocessor, postprocessor


def load_demo_init_states(hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5"):
    """Charge le premier état (init) de chaque démo Robomimic.

    On utilise ces init pour reset l'env Robosuite dans une distribution matchant le training,
    plutôt que la randomisation par défaut de Robosuite (qui produit des init out-of-distribution).
    """
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        init_states = np.stack([f["data"][d]["states"][0] for d in demos])
    print(f"Init states chargés : {init_states.shape[0]} démos, state_dim={init_states.shape[1]}", flush=True)
    return init_states


def eval_in_simulation(policy, preprocessor, postprocessor, device, n_eval_episodes=N_EVAL_EPISODES, save_videos=True):
    """Évalue la Diffusion Policy dans Robosuite Lift.

    Init de chaque épisode = état initial d'une démo (samplé aléatoirement parmi les 200),
    pour rester dans la distribution training. La randomisation par défaut de rs.make("Lift")
    produit des positions cube/eef hors training et le modèle fait n'importe quoi.
    """
    import robosuite as rs

    print(f"\nÉval {n_eval_episodes} épisodes sur Robosuite Lift...", flush=True)
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names="agentview",
        camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
        control_freq=20, reward_shaping=False,
    )

    init_states = load_demo_init_states()
    rng = np.random.default_rng(seed=1000)

    results = []
    all_frames = []
    t_start = time.time()

    for ep in range(n_eval_episodes):
        ep_t0 = time.time()
        env.reset()
        # Override avec un init state de démo (distribution training)
        demo_idx = int(rng.integers(0, init_states.shape[0]))
        env.sim.set_state_from_flattened(init_states[demo_idx])
        env.sim.forward()
        obs = env._get_observations()
        policy.reset()   # clean l'action queue interne (chunk de 8 actions)

        frames = []
        max_z = 0
        success = False
        ep_reward = 0

        for step in range(MAX_STEPS):
            if step % 50 == 0 and step > 0:
                print(f"    ep {ep+1:3d} step {step:3d}/{MAX_STEPS} (max_z={max_z:.3f})", flush=True)
            # Video frame (haute résolution pour visu)
            if save_videos and ep < N_VIDEOS:
                frame_hd = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
                frames.append(frame_hd)

            # Image pour le policy (96x96 comme au training)
            img = env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]
            state_np = build_state_vector(obs).astype(np.float32)
            state_t = torch.tensor(state_np)

            obs_dict = {
                "observation.image": img_t.unsqueeze(0).to(device),
                "observation.state": state_t.unsqueeze(0).to(device),
            }

            # Preprocessor : normalise image (uint8 → float 0-1) + normalise state via stats dataset
            obs_dict = preprocessor(obs_dict)
            # Le policy gère son propre queue d'actions (diffusion → 16 actions, exécute 8, re-prédit)
            with torch.no_grad():
                action_norm = policy.select_action(obs_dict)
            # Postprocessor : dé-normalise l'action (du range [-1, 1] normalisé vers le range réel)
            action = postprocessor(action_norm).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if "cube_pos" in obs:
                max_z = max(max_z, obs["cube_pos"][2])
            if env._check_success():
                success = True
            if done:
                break

        results.append({"reward": ep_reward, "success": success, "max_z": max_z})
        ep_time = time.time() - ep_t0
        elapsed = time.time() - t_start
        running_success = sum(r["success"] for r in results) / len(results)
        eta_min = (elapsed / (ep + 1)) * (n_eval_episodes - ep - 1) / 60
        print(f"  ep {ep + 1:3d}/{n_eval_episodes} : {'✓' if success else '✗'} "
              f"max_z={max_z:.3f} | running success={running_success:.0%} | "
              f"ep_time={ep_time:.1f}s | elapsed={elapsed/60:.1f}min | ETA={eta_min:.0f}min",
              flush=True)
        if frames and len(all_frames) < N_VIDEOS:
            all_frames.append(frames)

    env.close()

    n_success = sum(r["success"] for r in results)
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_max_z = float(np.mean([r["max_z"] for r in results]))
    elapsed = time.time() - t_start
    print(f"\n=== RÉSULTATS ===", flush=True)
    print(f"  Success : {n_success}/{n_eval_episodes} ({n_success/n_eval_episodes:.0%})", flush=True)
    print(f"  Avg max_z : {avg_max_z:.3f} m", flush=True)
    print(f"  Avg reward : {avg_reward:.2f}", flush=True)
    print(f"  Eval time : {elapsed:.0f}s ({elapsed/n_eval_episodes:.1f}s/ep)", flush=True)

    return {
        "success_rate": n_success / n_eval_episodes,
        "avg_reward": avg_reward,
        "avg_max_cube_z": avg_max_z,
        "eval_time_s": elapsed,
    }, all_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Chemin vers le dossier pretrained_model du checkpoint LeRobot")
    parser.add_argument("--n_eval", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--save_videos", action="store_true", default=True)
    args = parser.parse_args()

    EXPERIMENT_NAME = "lift/46_diffusion_official"
    DATASET = "robomimic/lift_ph"

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device)
    eval_results, all_frames = eval_in_simulation(policy, preprocessor, postprocessor, device,
                                                   n_eval_episodes=args.n_eval,
                                                   save_videos=args.save_videos)

    # Save videos + run_info dans notre format
    label = "lerobot_diffusion"
    run_dir = get_run_dir(EXPERIMENT_NAME, label)
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=20)

    # Garder une trace simple
    n_params = sum(p.numel() for p in policy.parameters())
    run_data = build_run_data(
        experiment=EXPERIMENT_NAME, architecture="LeRobot DiffusionPolicy (U-Net 1D + DDPM + ResNet18)",
        dataset=DATASET, n_episodes=200, n_params=n_params,
        size_mb=n_params * 4 / 1e6, seed=1000,
        train_losses=[], test_losses=[],   # pas accès aux losses depuis checkpoint
        train_time_s=0, epoch_times=[],
        flops_total_train=0, n_train_samples=9666, n_test_samples=0,
        inference_time_ms=eval_results["eval_time_s"] * 1000 / (args.n_eval * MAX_STEPS),
        flops_inference=0, frames_per_call=policy.config.n_action_steps,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_max_cube_z"],
        avg_reward=eval_results["avg_reward"],
        extra={
            "task": "Robomimic Lift",
            "checkpoint": args.checkpoint,
            "n_eval_episodes": args.n_eval,
            "policy_type": "diffusion (LeRobot official, lerobot-train CLI)",
            "training_steps": "15000",
        },
    )
    save_run(run_data)


if __name__ == "__main__":
    main()
