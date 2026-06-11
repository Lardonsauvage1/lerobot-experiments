"""Phase 5 — Variance de la MESURE 500-rollouts (différents jeux de départs).

Question : deux jeux de 500 départs (seeds différents) donnent-ils le même succès ?
On réévalue le checkpoint 10000 (dense) sur seed=1 et seed=2, à comparer avec
l'original seed=0 (= 30.4 %, dans rollouts_500.csv).

Sortie : results/runs/phase5_methodology/26_resnet34_dense/variance_500_seeds.csv
Tourne sur Mac (MPS). Lancer :
  nohup venv312/bin/python -u experiments/phase5_methodology/09_variance_500_seeds.py \
    > results/logs/phase5_methodology/run_09_variance500.log 2>&1 &
"""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv, math, time
from pathlib import Path
import numpy as np
import torch

from src import can_eval

RUN_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense")
CKPT_DIR = RUN_DIR / "checkpoints" / "010000"
OUT_CSV = RUN_DIR / "variance_500_seeds.csv"
STATES_DIR = Path("results/runs/can")
N_EVAL = 500
STEPS_INFER = 10
SEEDS = [1, 2]


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


def make_states(seed, n=N_EVAL):
    path = STATES_DIR / f"can_eval500_seed{seed}.npy"
    if path.exists():
        arr = np.load(path)
        print(f"[09] états seed={seed} chargés {arr.shape}", flush=True)
        return arr
    print(f"[09] génération {n} états Can seed={seed}...", flush=True)
    genenv = can_eval.make_env()
    np.random.seed(seed)
    states = []
    for i in range(n):
        genenv.reset()
        states.append(np.asarray(genenv.get_state()["states"]).copy())
        if (i + 1) % 100 == 0:
            print(f"[09]   {i + 1}/{n}", flush=True)
    arr = np.stack(states)
    np.save(path, arr)
    print(f"[09] sauvé {path} {arr.shape}", flush=True)
    return arr


def rollout_can(policy, pre, post, states, device, image_keys, chunk=50):
    per_all = []
    for i in range(0, len(states), chunk):
        env = can_eval.make_env()
        for ep in range(i, min(i + chunk, len(states))):
            obs = env.reset_to(dict(states=states[ep]))
            policy.reset()
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
            per_all.append(t_success is not None)
        try:
            env.env.close()
        except Exception:
            pass
    return per_all


def load_policy(device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    import json as _json
    ckpt_path = CKPT_DIR / "pretrained_model"
    ckpt = str(ckpt_path)
    # device_processor -> device du run (mps sur Mac)
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = ckpt_path / fn
        if not p.exists():
            continue
        cfg = _json.loads(p.read_text())
        changed = False
        for step in cfg.get("steps", []):
            if step.get("registry_name") == "device_processor" and step.get("config", {}).get("device") != device.type:
                step["config"]["device"] = device.type
                changed = True
        if changed:
            p.write_text(_json.dumps(cfg, indent=2))
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS_INFER
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def append_csv(row):
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[09] device={device}, checkpoint=010000", flush=True)
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    policy, pre, post = load_policy(device)
    print("[09] policy chargée", flush=True)
    print("[09] RAPPEL original seed=0 : 152/500 = 30.4% [26.5-34.6]", flush=True)
    for seed in SEEDS:
        states = make_states(seed)
        t0 = time.time()
        per = rollout_can(policy, pre, post, states, device, image_keys)
        elapsed = time.time() - t0
        k = sum(per); n = len(per)
        lo, hi = wilson_ci(k, n)
        append_csv({"seed": seed, "n_success": k, "n": n, "success_rate": k / n,
                    "ci95_low": lo, "ci95_high": hi, "elapsed_sec": elapsed})
        print(f"[09] seed={seed} : {k}/{n} = {k/n:.1%} [{lo:.1%}-{hi:.1%}] ({elapsed:.0f}s)", flush=True)
    print("[09] VARIANCE 500 SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
