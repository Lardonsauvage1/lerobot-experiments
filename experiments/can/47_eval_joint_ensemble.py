"""Éval JOINT avec TEMPORAL ENSEMBLING (façon ACT) — teste si ça stabilise la boucle fermée.

À chaque pas : on interroge la policy (chunk complet d'horizon ~15 actions prédites), on stocke,
et on EXÉCUTE la moyenne pondérée de toutes les prédictions passées qui visent le pas courant
(poids exp(-m·age)). C'est le mécanisme d'ACT contre l'accumulation d'erreurs. Éval-seule (pas de
réentraînement), sur le checkpoint 40k. À comparer au 40k sans ensembling (~12-15%).

Usage : venv312/bin/python -u experiments/can/47_eval_joint_ensemble.py --steps 40000 --n 50 [--m 0.01]
"""
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np, torch
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

STATES_PATH = "results/runs/can/can_eval500.npy"
HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
MAX_STEPS = 300; INFER_STEPS = 10; CHUNK = 50


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n
    return ((p+z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d,
            (p+z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d)


def state_joint(obs):
    return np.concatenate([np.asarray(obs["robot0_joint_pos"]).flatten(),
                           np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def build_joint_env(kp=50):
    import robomimic.utils.env_utils as EnvUtils, robomimic.utils.file_utils as FileUtils, robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_joint_pos", "robot0_gripper_qpos", "object"], "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5)
    arm = {"type": "JOINT_POSITION", "input_type": "absolute", "kp": kp, "damping_ratio": 1,
           "impedance_mode": "fixed", "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
           "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2,
           "input_max": 1, "input_min": -1, "output_max": 1, "output_min": -1, "gripper": {"type": "GRIP"}}
    env_meta["env_kwargs"]["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}
    return EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)


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
    """Renvoie tout l'horizon futur prédit (normalisé), shape (Hfut, A)."""
    od = dict(od)
    od.pop("action", None)  # comme select_action : ne pas pousser l'action (None) dans les files
    if policy.config.image_features:
        od[OBS_IMAGES] = torch.stack([od[k] for k in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(policy._queues, od)
    b = {k: torch.stack(list(policy._queues[k]), dim=1) for k in policy._queues if len(policy._queues[k]) > 0}
    gc = policy.diffusion._prepare_global_conditioning(b)
    acts = policy.diffusion.conditional_sample(b[OBS_STATE].shape[0], global_cond=gc)  # (1,horizon,A) normalisé
    start = policy.config.n_obs_steps - 1
    return acts[0, start:]  # (Hfut, A) tensor sur device, normalisé


def eval_ensemble(ckpt, states, device, kp, m, max_steps, no_ensemble=False):
    policy, pre, post = load(ckpt, device)
    succ = []; t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = build_joint_env(kp)
        for ep in range(i, min(i+CHUNK, len(states))):
            obs = env.reset_to({"states": states[ep]}); policy.reset()
            buf = {}  # query_time -> (Hfut, A) tensor normalisé
            ok = False
            for t in range(max_steps):
                images = {}
                for cam, key in IMAGE_KEYS.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                st = torch.from_numpy(state_joint(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                with torch.no_grad():
                    chunk_t = full_chunk(policy, od, device)  # (Hfut,A)
                if no_ensemble:
                    a_norm = chunk_t[0:1]  # action courante seule, pas de moyennage temporel (baseline)
                else:
                    buf[t] = chunk_t
                    # ensemble : toutes les prédictions visant le pas t
                    preds, ws = [], []
                    for tq, chunk in list(buf.items()):
                        age = t - tq
                        if 0 <= age < chunk.shape[0]:
                            preds.append(chunk[age]); ws.append(math.exp(-m * age))
                        elif age >= chunk.shape[0]:
                            del buf[tq]
                    w = torch.tensor(ws, device=device, dtype=preds[0].dtype); w = w / w.sum()
                    a_norm = (torch.stack(preds) * w[:, None]).sum(0, keepdim=True)  # (1,A) normalisé
                with torch.no_grad():
                    a = post(a_norm).squeeze(0).cpu().numpy().astype(np.float32)  # 8D joint brut
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
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kp", type=float, default=50)
    ap.add_argument("--m", type=float, default=0.01, help="poids ensemble exp(-m*age)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", default="results/runs/can/joint_r34_bigunet/ensemble_eval.csv")
    ap.add_argument("--ckpt", default=None, help="chemin pretrained_model direct (ex: modèle SWA) ; sinon depuis --steps")
    ap.add_argument("--no-ensemble", action="store_true", help="baseline : action courante seule, sans moyennage temporel")
    a = ap.parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)[: a.n]
    ck = a.ckpt or f"results/runs/can/joint_r34_bigunet/checkpoints/{a.steps:06d}/pretrained_model"
    mode = "SANS ensemble (baseline)" if a.no_ensemble else f"TEMPORAL ENSEMBLING m={a.m}"
    print(f"[47] {mode} joint ckpt={ck} n={a.n} device={device}", flush=True)
    res = eval_ensemble(ck, states, device, a.kp, a.m, a.max_steps, no_ensemble=a.no_ensemble)
    import csv
    out = Path(a.out); wh = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "m", *res.keys()])
        if wh: w.writeheader()
        w.writerow({"step": a.steps, "m": a.m, **res})
    print(f"[47] step={a.steps} ENSEMBLE: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
          f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s) — vs 40k sans ensemble ~12-15%", flush=True)
    print("[47] DONE", flush=True)


if __name__ == "__main__":
    main()
