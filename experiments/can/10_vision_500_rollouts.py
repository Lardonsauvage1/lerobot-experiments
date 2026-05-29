"""Éval rigoureuse à 500 rollouts des 3 modèles Can vision pure (02/06/08).

Les évals à 50 val donnent des IC95 ±13 pts -> on ne peut pas trancher.
500 rollouts -> IC95 ±~4 pts -> vraie comparaison.

États : 500 resets aléatoires de l'env Can figés (jamais entraînés).
Rollouts chunked (env recréé tous les 50 ép.) — anti-dégradation renderer.

Sortie : results/runs/can/vision_500.json
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src import can_eval

STATES_PATH = Path("results/runs/can/can_eval500.npy")
OUT = Path("results/runs/can/vision_500.json")
N_EVAL = 500
STEPS = 10
CHUNK = 50

MODELS = [
    {"name": "02_proprio (mono-cam)", "ckpt": "results/runs/can/02_proprio/checkpoints/010000/pretrained_model",
     "cams": ["agentview"], "image_keys": {"agentview": "observation.image"}},
    {"name": "06_proprio_wrist (shared)", "ckpt": "results/runs/can/06_proprio_wrist/checkpoints/010000/pretrained_model",
     "cams": ["agentview", "robot0_eye_in_hand"],
     "image_keys": {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "08_proprio_wrist_sep (séparé)", "ckpt": "results/runs/can/08_proprio_wrist_sep/checkpoints/010000/pretrained_model",
     "cams": ["agentview", "robot0_eye_in_hand"],
     "image_keys": {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}},
]


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


def make_eval_states(n=N_EVAL, seed=0):
    """500 états initiaux figés = resets aléatoires de l'env Can (jamais entraînés).
    Env JETABLE pour générer (pas d'impact sur l'env de rollout — bug renderer phase 4)."""
    if STATES_PATH.exists():
        arr = np.load(STATES_PATH)
        print(f"États Can chargés : {arr.shape} ({STATES_PATH})", flush=True)
        return arr
    print(f"Génération de {n} états Can (env jetable, seed={seed})...", flush=True)
    genenv = can_eval.make_env()
    np.random.seed(seed)
    states = []
    for i in range(n):
        genenv.reset()
        states.append(np.asarray(genenv.get_state()["states"]).copy())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    arr = np.stack(states)
    STATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(STATES_PATH, arr)
    print(f"Sauvé : {STATES_PATH} {arr.shape}", flush=True)
    return arr


def rollout_can(policy, pre, post, states, device, image_keys, chunk=CHUNK):
    """Rollout chunked. Recrée l'env tous les `chunk` épisodes (anti-dégradation renderer)."""
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
        done = sum(e["success"] for e in per_all)
        print(f"  [chunk {i}-{i + chunk}] cumul {done}/{len(per_all)} = {done / len(per_all):.1%}", flush=True)
    return per_all, time.time() - t0


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = make_eval_states()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def load(ckpt):
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        policy.diffusion.num_inference_steps = STEPS
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return policy, pre, post

    results = []
    if OUT.exists():
        results = json.loads(OUT.read_text()).get("results", [])
    done_names = {r["name"] for r in results}

    for model in MODELS:
        if model["name"] in done_names:
            print(f"\n{model['name']} déjà fait, skip", flush=True); continue
        print(f"\n##### {model['name']} — {model['ckpt']} #####", flush=True)
        policy, pre, post = load(model["ckpt"])
        per, elapsed = rollout_can(policy, pre, post, states, device, model["image_keys"])
        k = sum(e["success"] for e in per); n = len(per)
        lo, hi = wilson_ci(k, n)
        t_succs = [e["t_success"] for e in per if e["success"]]
        t_med = float(np.median(t_succs)) if t_succs else None
        rec = {"name": model["name"], "ckpt": model["ckpt"], "cams": model["cams"],
               "n": n, "n_success": k, "success_rate": k / n, "ci95": [lo, hi],
               "t_success_median": t_med, "elapsed_min": elapsed / 60}
        results.append(rec)
        print(f"\n=> {model['name']} : {k}/{n} = {k/n:.1%} [{lo:.1%}, {hi:.1%}]  t_med={t_med}  ({elapsed/60:.1f}min)", flush=True)
        OUT.write_text(json.dumps({"n_eval": N_EVAL, "steps": STEPS, "results": results}, indent=2))
        del policy

    print("\n=== RÉCAP 500 ROLLOUTS ===", flush=True)
    for r in results:
        print(f"  {r['name']:40s}: {r['success_rate']*100:5.1f}% [{r['ci95'][0]*100:.1f}-{r['ci95'][1]*100:.1f}]  t={r['t_success_median']}")
    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
