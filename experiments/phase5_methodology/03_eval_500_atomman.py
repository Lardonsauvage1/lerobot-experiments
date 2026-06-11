"""Phase 5 — Worker atomman : poll les ckpts qui arrivent, rollout 500 sur ceux multiples de 1000.

Tourne sur atomman (CPU 22 threads). Polle results/runs/phase5_methodology/26_resnet34_dense/checkpoints/
À chaque nouveau ckpt step % 1000 == 0 -> rollout 500 figés (can_eval500.npy), append à rollouts_500.csv.
Termine quand 30 ckpts traités (step 1000 à 30000) OU sentinel /tmp/stop_03 créé.

Sortie : results/runs/phase5_methodology/26_resnet34_dense/rollouts_500.csv
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from src import can_eval

RUN_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense")
CKPT_ROOT = RUN_DIR / "checkpoints"
OUT_CSV = RUN_DIR / "rollouts_500.csv"
STATES_PATH = Path("results/runs/can/can_eval500.npy")
STEPS_INFER = 10
EVAL_INTERVAL = 1000   # tous les 1000 step ckpts


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


def rollout_can(policy, pre, post, states, device, image_keys, chunk=50):
    """Reprend lift_eval.rollout_eval_chunked pattern : recrée l'env tous les chunk ép."""
    per_all = []
    t0 = time.time()
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
            per_all.append({"success": t_success is not None, "t_success": t_success})
        try:
            env.env.close()
        except Exception:
            pass
    return per_all, time.time() - t0


def eval_one(ckpt_dir, states, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    ckpt_path = Path(ckpt_dir) / "pretrained_model"
    ckpt = str(ckpt_path)
    # Patch in-place les JSON: ckpt vient de Mac MPS, atomman = CPU.
    # Plus robuste que overrides= (qui semble pas marcher sur atomman).
    import json as _json
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = ckpt_path / fn
        if not p.exists():
            continue
        cfg = _json.loads(p.read_text())
        changed = False
        for step in cfg.get("steps", []):
            if step.get("registry_name") == "device_processor":
                if step.get("config", {}).get("device") != "cpu":
                    step["config"]["device"] = "cpu"
                    changed = True
        if changed:
            p.write_text(_json.dumps(cfg, indent=2))
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS_INFER
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    per, elapsed = rollout_can(policy, pre, post, states, device, image_keys)
    k = sum(e["success"] for e in per); n = len(per)
    lo, hi = wilson_ci(k, n)
    t_succs = [e["t_success"] for e in per if e["success"]]
    t_med = float(np.median(t_succs)) if t_succs else None
    return {"n_success": k, "n": n, "success_rate": k / n,
            "ci95_low": lo, "ci95_high": hi,
            "t_success_median": t_med, "elapsed_sec": elapsed}


def load_done_steps():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(int(row["step"]))
    return done


def append_csv(row):
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    print(f"[03] start — poll {CKPT_ROOT}", flush=True)
    device = torch.device("cpu")
    states = np.load(STATES_PATH)
    print(f"[03] {states.shape[0]} init states loaded", flush=True)
    done = load_done_steps()
    print(f"[03] {len(done)} steps déjà évalués : {sorted(done)}", flush=True)

    last_attempt = 0
    while True:
        if Path("/tmp/stop_03").exists():
            print(f"[03] stop sentinel détecté"); break

        if not CKPT_ROOT.exists():
            time.sleep(60); continue

        # Lister ckpts dispo et multiples de EVAL_INTERVAL
        avail = []
        for p in sorted(CKPT_ROOT.iterdir()):
            if not p.is_dir() or not p.name.isdigit():
                continue
            step = int(p.name)
            if step % EVAL_INTERVAL == 0 and step not in done:
                # vérifie complétude (seulement pretrained_model — training_state
                # n'est pas poussé par 02_push pour économiser bande passante)
                if (p / "pretrained_model" / "model.safetensors").exists():
                    avail.append((step, p))

        if not avail:
            # Si rien de nouveau et qu'on a tout (30 ckpts), on sort
            if len(done) >= 30:
                print(f"[03] tous les 30 ckpts traités, fin"); break
            time.sleep(60); continue

        # Traite le plus petit step en premier (ordre chronologique)
        step, p = avail[0]
        print(f"[03] eval ckpt step={step}", flush=True)
        try:
            t0 = time.time()
            res = eval_one(p, states, device)
            row = {"step": step, **res}
            append_csv(row)
            done.add(step)
            print(f"[03] step={step} : {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] elapsed={res['elapsed_sec']:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"[03] ⚠ erreur step={step} : {e}", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
