"""Eval comparative des 4 checkpoints du run 46 — répond à "entraîner plus longtemps a-t-il aidé ?".

Pour chaque checkpoint (003K, 006K, 009K, 012K), lance N épisodes (Robomimic env + sign fix)
et mesure le success rate. Sauve un JSON + une figure combinée :
- Train loss (axe gauche, log) — parsé depuis le log lerobot-train
- Success rate (axe droit) à chaque checkpoint step
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import time
import json
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import re
from pathlib import Path
from src.lift_data import STATE_KEYS

CHECKPOINT_ROOT = Path("results/runs/lift/46_diffusion_official/checkpoints")
HDF5_PATH = "data_cache/robomimic_lift_ph/low_dim_v15.hdf5"
LOG_PATH = Path("results/logs/lift/run_46_diffusion.log")
OUT_DIR = Path("results/runs/lift/46_diffusion_official")
N_EVAL_PER_CKPT = 20
MAX_STEPS = 200
IMAGE_SIZE = 96
NUM_INFERENCE_STEPS = 10
CHECKPOINT_STEPS = [3000, 6000, 9000, 12000]


def patch_controller_config(env_meta):
    old_ctrl = env_meta['env_kwargs']['controller_configs']
    new_arm_ctrl = dict(old_ctrl)
    if 'damping' in new_arm_ctrl: new_arm_ctrl['damping_ratio'] = new_arm_ctrl.pop('damping')
    if 'damping_limits' in new_arm_ctrl: new_arm_ctrl['damping_ratio_limits'] = new_arm_ctrl.pop('damping_limits')
    new_arm_ctrl.setdefault('input_type', 'delta')
    new_arm_ctrl.setdefault('input_ref_frame', 'base')
    new_arm_ctrl['gripper'] = {'type': 'GRIP'}
    env_meta['env_kwargs']['controller_configs'] = {'type': 'BASIC', 'body_parts': {'right': new_arm_ctrl}}


def fix_obs_sign(obs):
    obs = dict(obs)
    if 'object' in obs:
        obj = np.asarray(obs['object']).copy()
        obj[7:10] = -obj[7:10]
        obs['object'] = obj
    return obs


def build_state_vector(obs):
    parts = [np.asarray(obs[k]).flatten() for k in STATE_KEYS]
    return np.concatenate(parts).astype(np.float32)


def make_env():
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {'low_dim': list(STATE_KEYS), 'rgb': []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5_PATH)
    patch_controller_config(env_meta)
    return EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)


def load_init_states():
    with h5py.File(HDF5_PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        return np.stack([f["data"][d]["states"][0] for d in demos])


def eval_checkpoint(step: int, env, init_states, rng, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    ckpt = str(CHECKPOINT_ROOT / f"{step:06d}" / "pretrained_model")
    print(f"\n=== Checkpoint step {step} ===", flush=True)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    n_success = 0
    max_zs = []
    t_start = time.time()
    for ep in range(N_EVAL_PER_CKPT):
        demo_idx = int(rng.integers(0, init_states.shape[0]))
        obs = env.reset_to(dict(states=init_states[demo_idx]))
        obs = fix_obs_sign(obs)
        policy.reset()
        max_z = 0; success = False
        for step_i in range(MAX_STEPS):
            img = env.env.sim.render(height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            state_t = torch.from_numpy(build_state_vector(obs))
            obs_dict = {"observation.image": img_t.unsqueeze(0).to(device),
                        "observation.state": state_t.unsqueeze(0).to(device)}
            obs_dict = pre(obs_dict)
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = post(a).squeeze(0).cpu().numpy()
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            obs, _, done, _ = env.step(a)
            obs = fix_obs_sign(obs)
            cube_z = float(np.asarray(obs['object'])[2])
            max_z = max(max_z, cube_z)
            if env.is_success()["task"]:
                success = True
            if done: break
        n_success += int(success)
        max_zs.append(max_z)
        running = n_success / (ep + 1)
        print(f"  ep {ep+1:2d}/{N_EVAL_PER_CKPT} : {'✓' if success else '✗'} max_z={max_z:.3f} | running={running:.0%}", flush=True)

    success_rate = n_success / N_EVAL_PER_CKPT
    elapsed = time.time() - t_start
    print(f"  → step {step}: {n_success}/{N_EVAL_PER_CKPT} = {success_rate:.0%}  (avg max_z={np.mean(max_zs):.3f}, {elapsed:.0f}s)", flush=True)
    del policy, pre, post
    return {"step": step, "success_rate": success_rate, "n_success": n_success, "n_total": N_EVAL_PER_CKPT,
            "avg_max_z": float(np.mean(max_zs)), "max_zs": [float(z) for z in max_zs]}


def parse_train_loss(path: Path):
    RX = re.compile(r"loss:([\d.eE+-]+)")
    if not path.exists(): return [], []
    text = path.read_text(errors="ignore")
    steps, losses = [], []
    idx = 0
    for line in text.split("\n"):
        m = RX.search(line)
        if m:
            idx += 1
            steps.append(idx * 200)
            losses.append(float(m.group(1)))
    return steps, losses


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    env = make_env()
    init_states = load_init_states()
    rng = np.random.default_rng(seed=1000)

    results = []
    for step in CHECKPOINT_STEPS:
        results.append(eval_checkpoint(step, env, init_states, rng, device))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  step {r['step']:6d}: success {r['n_success']:2d}/{r['n_total']} = {r['success_rate']:.0%}")

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "checkpoint_eval.json"
    with open(json_path, "w") as f:
        json.dump({"n_eval_per_checkpoint": N_EVAL_PER_CKPT, "results": results}, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Plot
    train_steps, train_losses = parse_train_loss(LOG_PATH)
    eval_steps = [r["step"] for r in results]
    eval_rates = [r["success_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    if train_steps:
        ax1.plot(train_steps, train_losses, color="#1f77b4", linewidth=1.3, label="Train loss (noise prediction MSE)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Train loss (log scale)", color="#1f77b4")
    ax1.set_yscale("log")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eval_steps, [r * 100 for r in eval_rates], "o-", color="#d62728", markersize=10, linewidth=2,
             label=f"Eval success rate ({N_EVAL_PER_CKPT} ep / checkpoint)")
    ax2.set_ylabel("Success rate (%)", color="#d62728")
    ax2.set_ylim(-5, 105)
    ax2.tick_params(axis='y', labelcolor="#d62728")
    for s, r in zip(eval_steps, eval_rates):
        ax2.annotate(f"{r:.0%}", (s, r * 100), textcoords="offset points", xytext=(0, 12),
                     ha="center", color="#d62728", fontweight="bold")

    fig.suptitle("Run 46 — Diffusion Policy : train loss vs eval success across checkpoints", fontsize=12)
    fig.tight_layout()
    plot_path = OUT_DIR / "checkpoint_comparison.png"
    fig.savefig(plot_path, dpi=110)
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
