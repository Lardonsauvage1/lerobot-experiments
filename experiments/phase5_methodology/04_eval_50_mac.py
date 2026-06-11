"""Phase 5 — Worker Mac : rollout 50 (val set) sur les ckpts sauvés tous les 200 steps.

Lancer APRÈS la fin du train (Mac MPS libéré). Itère sur les ckpts dispo,
fait rollout sur les 50 ép. val à chaque ckpt step % 200 == 0.

Sortie : results/runs/phase5_methodology/26_resnet34_dense/rollouts_50.csv
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
OUT_CSV = RUN_DIR / "rollouts_50.csv"
STEPS_INFER = 10
EVAL_INTERVAL = 200    # tous les 200 step ckpts


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


def rollout_50(policy, pre, post, init_states, device):
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    env = can_eval.make_env()
    per = []
    t0 = time.time()
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
        per.append({"success": t_success is not None, "t_success": t_success})
    return per, time.time() - t0


def eval_one(ckpt_dir, init_states, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    ckpt = str(ckpt_dir / "pretrained_model")
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS_INFER
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    per, elapsed = rollout_50(policy, pre, post, init_states, device)
    k = sum(e["success"] for e in per); n = len(per)
    lo, hi = wilson_ci(k, n)
    t_succs = [e["t_success"] for e in per if e["success"]]
    t_med = float(np.median(t_succs)) if t_succs else None
    return {"n_success": k, "n": n, "success_rate": k / n,
            "ci95_low": lo, "ci95_high": hi,
            "t_success_median": t_med, "elapsed_sec": elapsed}


def load_done():
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
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = can_eval.load_or_make_split()["val"]
    init_states = can_eval.load_init_states(val_idx)
    print(f"[04] {len(init_states)} val states", flush=True)
    done = load_done()
    print(f"[04] {len(done)} steps déjà évalués", flush=True)

    while True:
        avail = []
        if CKPT_ROOT.exists():
            for p in sorted(CKPT_ROOT.iterdir()):
                if not p.is_dir() or not p.name.isdigit():
                    continue
                step = int(p.name)
                if step % EVAL_INTERVAL == 0 and step not in done:
                    if (p / "pretrained_model").exists() and (p / "training_state").exists():
                        avail.append((step, p))

        if not avail:
            if Path("/tmp/stop_04").exists():
                print(f"[04] stop sentinel, fin"); break
            time.sleep(60); continue

        step, p = avail[0]
        print(f"[04] eval ckpt step={step}", flush=True)
        try:
            res = eval_one(p, init_states, device)
            row = {"step": step, **res}
            append_csv(row)
            done.add(step)
            print(f"[04] step={step} : {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] elapsed={res['elapsed_sec']:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"[04] ⚠ erreur step={step} : {e}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
