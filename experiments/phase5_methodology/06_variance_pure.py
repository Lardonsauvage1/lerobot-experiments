"""Phase 5 — Variance pure du rollout 50 : 5 ckpts clés × 5 répétitions.

Mesure la variance intrinsèque du rollout 50 lui-même (pas la variance entre ckpts).
Pour chaque ckpt clé, fait 5 fois rollout_50 -> distribution du succès.
À lancer APRÈS 04_eval_50_mac.py terminé.

Sortie : results/runs/phase5_methodology/26_resnet34_dense/variance_pure.csv
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from src import can_eval

RUN_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense")
CKPT_ROOT = RUN_DIR / "checkpoints"
OUT_CSV = RUN_DIR / "variance_pure.csv"
STEPS_INFER = 10
N_REPS = 5

# Ckpts clés à étudier (steps)
KEY_STEPS = [5000, 10000, 15000, 20000, 30000]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def rollout_50(policy, pre, post, init_states, device, seed=None):
    """rollout 50 avec env reset to figés. Le seed n'affecte que la diffusion sampling.
    Le init state est figé pour chaque ép. -> la variance vient de la stochasticité du sampling."""
    if seed is not None:
        torch.manual_seed(seed)
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    env = can_eval.make_env()
    per = []
    for ep in range(len(init_states)):
        obs = env.reset_to(dict(states=init_states[ep])); policy.reset()
        t_success = None
        for step_i in range(can_eval.MAX_STEPS):
            images = {}
            for cam, key in image_keys.items():
                img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                images[key] = img_t.unsqueeze(0).to(device)
            state_t = torch.from_numpy(state_proprio(obs))
            obs_dict = pre({**images, "observation.state": state_t.unsqueeze(0).to(device)})
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs = env.step(a)[0]
            if env.is_success()["task"]:
                t_success = step_i; break
        per.append({"success": t_success is not None})
    return per


def load_done():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((int(row["step"]), int(row["rep"])))
    return done


def append_csv(row):
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = can_eval.load_or_make_split()["val"]
    init_states = can_eval.load_init_states(val_idx)
    print(f"[06] device={device}, {len(init_states)} val states", flush=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    done = load_done()
    print(f"[06] {len(done)} (step, rep) déjà fait", flush=True)

    for step in KEY_STEPS:
        ckpt_dir = CKPT_ROOT / f"{step:06d}"
        if not (ckpt_dir / "pretrained_model").exists():
            print(f"[06] ⚠ ckpt step={step} absent, skip"); continue
        ckpt = str(ckpt_dir / "pretrained_model")
        try:
            policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
            policy.diffusion.num_inference_steps = STEPS_INFER
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
                to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
            for rep in range(N_REPS):
                if (step, rep) in done:
                    continue
                t0 = time.time()
                per = rollout_50(policy, pre, post, init_states, device, seed=42 + rep)
                k = sum(e["success"] for e in per); n = len(per)
                lo, hi = wilson_ci(k, n)
                row = {"step": step, "rep": rep,
                       "n_success": k, "n": n, "success_rate": k / n,
                       "ci95_low": lo, "ci95_high": hi,
                       "elapsed_sec": time.time() - t0}
                append_csv(row)
                print(f"[06] step={step} rep={rep}: {k}/{n} = {k/n:.0%} ({row['elapsed_sec']:.0f}s)", flush=True)
            del policy
        except Exception as e:
            print(f"[06] ⚠ erreur step={step} : {e}", flush=True)
    print(f"[06] DONE")


if __name__ == "__main__":
    main()
