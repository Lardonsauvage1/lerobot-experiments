"""Éval CARTÉSIEN (run31, OSC) avec/sans TEMPORAL ENSEMBLING — pour tester si l'ensembling
aide aussi le cartésien (déjà ~94,8% au plafond) ou pas (mécanisme = lissage du jitter).

Jumeau de 47_eval_joint_ensemble.py mais espace CARTÉSIEN : env OSC (can_eval), state = eef
(proprio 9D), action 7D OSC clippée [-1,1]. --no-ensemble => baseline (select_action normal).

Usage : venv312/bin/python -u experiments/can/48_eval_cartesian_ensemble.py --steps 40000 --n 30 --m 0.01 [--no-ensemble]
"""
import argparse, csv, math, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np, torch
from src import can_eval
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.policies.utils import populate_queues

STATES_PATH = "results/runs/can/can_eval500.npy"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
MAX_STEPS = 300; INFER_STEPS = 10; CHUNK = 50
RUN = "results/runs/can/31_proprio_birdview_r34_bigunet"


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n
    return ((p+z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d,
            (p+z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d)


def state_proprio(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def load(ckpt, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = INFER_STEPS
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def full_chunk(policy, od, device):
    od = dict(od); od.pop("action", None)
    if policy.config.image_features:
        od[OBS_IMAGES] = torch.stack([od[k] for k in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(policy._queues, od)
    b = {k: torch.stack(list(policy._queues[k]), dim=1) for k in policy._queues if len(policy._queues[k]) > 0}
    gc = policy.diffusion._prepare_global_conditioning(b)
    acts = policy.diffusion.conditional_sample(b[OBS_STATE].shape[0], global_cond=gc)
    start = policy.config.n_obs_steps - 1
    return acts[0, start:]


def eval_run(ckpt, states, device, m, ensemble, max_steps):
    policy, pre, post = load(ckpt, device)
    succ = []; t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = can_eval.make_env()
        for ep in range(i, min(i+CHUNK, len(states))):
            obs = env.reset_to({"states": states[ep]}); policy.reset()
            buf = {}; ok = False
            for t in range(max_steps):
                images = {}
                for cam, key in IMAGE_KEYS.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
                st = torch.from_numpy(state_proprio(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                if ensemble:
                    with torch.no_grad():
                        buf[t] = full_chunk(policy, od, device)
                    preds, ws = [], []
                    for tq, chunk in list(buf.items()):
                        age = t - tq
                        if 0 <= age < chunk.shape[0]:
                            preds.append(chunk[age]); ws.append(math.exp(-m*age))
                        elif age >= chunk.shape[0]:
                            del buf[tq]
                    w = torch.tensor(ws, device=device, dtype=preds[0].dtype); w = w/w.sum()
                    a_norm = (torch.stack(preds)*w[:, None]).sum(0, keepdim=True)
                    with torch.no_grad():
                        a = post(a_norm).squeeze(0).cpu().numpy()
                else:
                    with torch.no_grad():
                        a = post(policy.select_action(od)).squeeze(0).cpu().numpy()
                a = np.clip(a, -1.0, 1.0).astype(np.float32)  # OSC cartésien
                obs = env.step(a)[0]
                if env.is_success()["task"]: ok = True; break
            succ.append(ok)
        try: env.env.close()
        except Exception: pass
    k = sum(succ); n = len(succ); lo, hi = wilson(k, n)
    return {"n_success": k, "n": n, "success_rate": k/n, "ci95_low": lo, "ci95_high": hi, "elapsed_sec": time.time()-t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--m", type=float, default=0.01)
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", default=f"{RUN}/cartesian_ensemble.csv")
    a = ap.parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)[: a.n]
    ck = f"{RUN}/checkpoints/{a.steps:06d}/pretrained_model"
    ens = not a.no_ensemble
    mode = f"ensemble(m={a.m})" if ens else "baseline"
    print(f"[48] CARTÉSIEN {mode} step={a.steps} n={a.n} device={device}", flush=True)
    res = eval_run(ck, states, device, a.m, ens, a.max_steps)
    out = Path(a.out); wh = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "mode", "m", *res.keys()])
        if wh: w.writeheader()
        w.writerow({"step": a.steps, "mode": "ensemble" if ens else "baseline", "m": a.m if ens else "", **res})
    print(f"[48] step={a.steps} {mode}: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
          f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
    print("[48] DONE", flush=True)


if __name__ == "__main__":
    main()
