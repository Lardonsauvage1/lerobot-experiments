"""Éval ARTICULAIRE par checkpoint d'un run birdview joint (DiffusionPolicy).

Jumeau de 32_eval_dense_birdview.py mais en espace MOTEUR :
  - env contrôlé en JOINT_POSITION ABSOLU (kp=50, validé étape 0 : replay 10/10)
  - state = robot0_joint_pos(7) + gripper_qpos(2)  (9D joint)
  - action = 8D [7 cibles joints absolues (rad), gripper] feedée TELLE QUELLE
    (PAS de clip [-1,1] : ce sont des radians, pas du OSC normalisé ; gripper géré par GRIP)
Mêmes états figés can_eval500.npy que le cartésien → succès comparable à run 31 (94,8 %@500).

Usage : venv312/bin/python -u experiments/can/37_eval_joint_birdview.py \
  --run-dir results/runs/can/<run_joint> --steps "" --n 50 --kp 50 \
  --out results/runs/can/<run_joint>/rollouts_50.csv
"""
import argparse, csv, math, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np
import torch

STATES_PATH = "results/runs/can/can_eval500.npy"
HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
MAX_STEPS = 300
INFER_STEPS = 10
CHUNK = 50


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    return ((p + z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d,
            (p + z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d)


def state_joint(obs):
    return np.concatenate([np.asarray(obs["robot0_joint_pos"]).flatten(),
                           np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def build_joint_env(kp):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_joint_pos", "robot0_gripper_qpos", "object"], "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5)
    arm = {"type": "JOINT_POSITION", "input_type": "absolute",
           "kp": kp, "damping_ratio": 1, "impedance_mode": "fixed",
           "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
           "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2,
           "input_max": 1, "input_min": -1, "output_max": 1, "output_min": -1,
           "gripper": {"type": "GRIP"}}
    env_meta["env_kwargs"]["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)


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


def eval_ckpt(ckpt, states, device, kp, max_steps, delta=False):
    policy, pre, post = load(ckpt, device)
    succ = []; t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = build_joint_env(kp)
        for ep in range(i, min(i + CHUNK, len(states))):
            obs = env.reset_to({"states": states[ep]}); policy.reset(); ok = False
            for _ in range(max_steps):
                images = {}
                for cam, key in IMAGE_KEYS.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                st = torch.from_numpy(state_joint(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                with torch.no_grad():
                    a = policy.select_action(od)
                a = post(a).squeeze(0).cpu().numpy().astype(np.float32)  # 8D joint, PAS de clip [-1,1]
                if delta:
                    # A predit un DELTA articulaire -> cible absolue = jpos_courant + delta (gripper reste absolu)
                    a[:7] = np.asarray(obs["robot0_joint_pos"]).flatten()[:7] + a[:7]
                obs = env.step(a)[0]
                if env.is_success()["task"]:
                    ok = True; break
            succ.append(ok)
        try: env.env.close()
        except Exception: pass
    k = sum(succ); n = len(succ); lo, hi = wilson_ci(k, n)
    return {"n_success": k, "n": n, "success_rate": k / n, "ci95_low": lo, "ci95_high": hi,
            "elapsed_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", default="", help="liste virgules ; vide -> tous les ckpts")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kp", type=float, default=50)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", action="store_true", help="action prédite = DELTA articulaire (cible = jpos+delta)")
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)[: a.n]
    ckroot = Path(a.run_dir) / "checkpoints"
    steps = [int(s) for s in a.steps.split(",")] if a.steps.strip() else \
            sorted(int(p.name) for p in ckroot.glob("[0-9]" * 6))
    out = Path(a.out); done = set()
    if out.exists():
        for r in csv.DictReader(open(out)): done.add(int(r["step"]))
    print(f"[37] JOINT eval, device={device}, kp={a.kp}, {len(steps)} ckpts, n={a.n}, faits={sorted(done)}", flush=True)
    for s in steps:
        if s in done:
            print(f"[37] step {s}: déjà fait, skip", flush=True); continue
        ck = ckroot / f"{s:06d}" / "pretrained_model"
        if not (ck / "model.safetensors").exists():
            print(f"[37] step {s}: absent, skip", flush=True); continue
        try:
            res = eval_ckpt(str(ck), states, device, a.kp, a.max_steps, delta=a.delta)
            wh = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", *res.keys()])
                if wh: w.writeheader()
                w.writerow({"step": s, **res})
            print(f"[37] step={s}: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
        except Exception as e:
            print(f"[37] erreur step={s}: {e}", flush=True)
    print("[37] DONE", flush=True)


if __name__ == "__main__":
    main()
