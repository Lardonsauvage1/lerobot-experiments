"""Eval Diffusion Policy v6 — utilise Robomimic env + sign flip pour matcher training.

Fixes accumulés depuis la v5 :
1. num_inference_steps = 10 (DDIM, vs DDPM 100 par défaut)
2. PolicyProcessorPipeline applique normalize/unnormalize
3. Reset depuis init state d'une démo HDF5 (distribution training)
4. Utilise Robomimic env wrapper (matche Robosuite 1.4.1 conventions)
5. SIGN FLIP sur object[7:10] (différence de convention 1.4 → 1.5 sur gripper↔cube relatif)
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import time
import torch
import numpy as np
import h5py
from pathlib import Path

from src.lift_data import STATE_KEYS
from src.tracker import build_run_data, save_run, save_eval_videos, get_run_dir

CHECKPOINT = "results/runs/lift/46_diffusion_official/checkpoints/012000/pretrained_model"
HDF5_PATH = "data_cache/robomimic_lift_ph/low_dim_v15.hdf5"
N_EVAL_EPISODES = 200
IMAGE_SIZE = 96
MAX_STEPS = 200
NUM_INFERENCE_STEPS = 10
N_VIDEOS = 3


def patch_controller_config(env_meta):
    """Convert old Robosuite 1.4 controller config to Robosuite 1.5 composite format."""
    old_ctrl = env_meta['env_kwargs']['controller_configs']
    new_arm_ctrl = dict(old_ctrl)
    if 'damping' in new_arm_ctrl:
        new_arm_ctrl['damping_ratio'] = new_arm_ctrl.pop('damping')
    if 'damping_limits' in new_arm_ctrl:
        new_arm_ctrl['damping_ratio_limits'] = new_arm_ctrl.pop('damping_limits')
    new_arm_ctrl.setdefault('input_type', 'delta')
    new_arm_ctrl.setdefault('input_ref_frame', 'base')
    new_arm_ctrl['gripper'] = {'type': 'GRIP'}
    env_meta['env_kwargs']['controller_configs'] = {
        'type': 'BASIC',
        'body_parts': {'right': new_arm_ctrl},
    }


def fix_obs_sign_convention(obs):
    """Robosuite 1.5 stocke object[7:10] = cube-eef ; Robomimic 1.4 HDF5 a eef-cube.
    On flip pour matcher la convention du dataset training."""
    obs = dict(obs)
    if 'object' in obs:
        obj = np.asarray(obs['object']).copy()
        obj[7:10] = -obj[7:10]
        obs['object'] = obj
    return obs


def build_state_vector(obs):
    parts = []
    for k in STATE_KEYS:
        parts.append(np.asarray(obs[k]).flatten())
    return np.concatenate(parts).astype(np.float32)


def make_env_robomimic():
    """Crée un env Robomimic-wrapped (matche format HDF5)."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs({
        'obs': {
            'low_dim': list(STATE_KEYS),
            'rgb': [],
        }
    })

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5_PATH)
    patch_controller_config(env_meta)
    # On veut images aussi (camera_obs interne désactivé, on rend manuellement via env.env.sim.render)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)
    return env


def load_demo_init_states():
    with h5py.File(HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        init_states = np.stack([f["data"][d]["states"][0] for d in demos])
    print(f"Init states chargés : {init_states.shape[0]} démos", flush=True)
    return init_states


def load_policy(device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    print(f"Loading Diffusion Policy from {CHECKPOINT}...", flush=True)
    policy = DiffusionPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy : {n_params:,} params, device={device}", flush=True)
    print(f"  num_inference_steps={policy.diffusion.num_inference_steps}", flush=True)
    return policy, preprocessor, postprocessor


def eval_loop(policy, preprocessor, postprocessor, env, init_states, device, n_eval_episodes, save_videos=True):
    rng = np.random.default_rng(seed=1000)
    results = []
    all_frames = []
    t_start = time.time()

    for ep in range(n_eval_episodes):
        ep_t0 = time.time()
        demo_idx = int(rng.integers(0, init_states.shape[0]))
        obs = env.reset_to(dict(states=init_states[demo_idx]))
        obs = fix_obs_sign_convention(obs)
        policy.reset()

        frames = []
        max_z = 0
        success = False
        ep_reward = 0

        for step in range(MAX_STEPS):
            if step % 50 == 0 and step > 0:
                print(f"    ep {ep+1:3d} step {step:3d}/{MAX_STEPS} (max_z={max_z:.3f})", flush=True)

            if save_videos and ep < N_VIDEOS:
                frame_hd = env.env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
                frames.append(frame_hd)

            img = env.env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            state_t = torch.from_numpy(build_state_vector(obs))

            obs_dict = {
                "observation.image": img_t.unsqueeze(0).to(device),
                "observation.state": state_t.unsqueeze(0).to(device),
            }
            obs_dict = preprocessor(obs_dict)
            with torch.no_grad():
                action_norm = policy.select_action(obs_dict)
            action = postprocessor(action_norm).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, reward, done, info = env.step(action)
            obs = fix_obs_sign_convention(obs)
            ep_reward += reward
            if 'object' in obs:
                cube_z = float(np.asarray(obs['object'])[2])
                max_z = max(max_z, cube_z)
            if env.is_success()["task"]:
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
    parser.add_argument("--n_eval", type=int, default=N_EVAL_EPISODES)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    policy, preprocessor, postprocessor = load_policy(device)
    env = make_env_robomimic()
    init_states = load_demo_init_states()

    eval_results, all_frames = eval_loop(policy, preprocessor, postprocessor, env, init_states,
                                          device, n_eval_episodes=args.n_eval)

    EXPERIMENT_NAME = "lift/46_diffusion_official"
    run_dir = get_run_dir(EXPERIMENT_NAME, "lerobot_diffusion_v6")
    if all_frames:
        save_eval_videos(all_frames, run_dir, fps=20)

    n_params = sum(p.numel() for p in policy.parameters())
    run_data = build_run_data(
        experiment=EXPERIMENT_NAME, architecture="LeRobot DiffusionPolicy (U-Net 1D + DDPM, eval v6 Robomimic env)",
        dataset="robomimic/lift_ph", n_episodes=args.n_eval, n_params=n_params,
        size_mb=n_params * 4 / 1e6, seed=1000,
        train_losses=[], test_losses=[],
        train_time_s=0, epoch_times=[],
        flops_total_train=0, n_train_samples=9666, n_test_samples=0,
        inference_time_ms=eval_results["eval_time_s"] * 1000 / (args.n_eval * MAX_STEPS),
        flops_inference=0, frames_per_call=policy.config.n_action_steps,
        success_rate=eval_results["success_rate"],
        avg_coverage=eval_results["avg_max_cube_z"],
        avg_reward=eval_results["avg_reward"],
        extra={"task": "Robomimic Lift", "checkpoint": CHECKPOINT,
               "training_steps": 12000, "num_inference_steps": NUM_INFERENCE_STEPS},
    )
    save_run(run_data)


if __name__ == "__main__":
    main()
