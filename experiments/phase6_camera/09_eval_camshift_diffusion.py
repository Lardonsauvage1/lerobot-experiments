"""Éval camshift pour DiffusionPolicy (gros modèle 61M) — MPS single-process.

Variante de 01_eval_camshift.py (qui charge un mini-CNN, CPU) pour un modèle DiffusionPolicy
birdview proprio (cartésien OSC). À chaque épisode, décale aléatoirement les 2 caméras
(translation + rotation, fixe pendant l'épisode), puis rollout. Sweep de niveaux → courbe
succès vs déplacement. Mêmes 500 états figés.

Usage :
  venv312/bin/python -u experiments/phase6_camera/09_eval_camshift_diffusion.py \
    --ckpt results/runs/phase6_camera/camaug_r34_bigunet/checkpoints/last/pretrained_model \
    --levels 0:0,2:2,5:5,10:10,15:15,20:20 --n 50 \
    --out results/runs/phase6_camera/camaug_r34_camshift.csv
"""
import argparse, csv, math, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np
import torch
from src import can_eval

STATES_PATH = "results/runs/can/can_eval500.npy"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
MAX_STEPS = 300
INFER_STEPS = 10
CHUNK = 50


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    return ((p + z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d,
            (p + z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d)


def state_proprio(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-9: return np.array([1.0, 0, 0, 0])
    axis = axis / n; h = angle / 2.0; s = math.sin(h)
    return np.array([math.cos(h), s*axis[0], s*axis[1], s*axis[2]])


def _quat_mult(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def _cam_id(model, name):
    try: return model.camera_name2id(name)
    except Exception:
        for i in range(model.ncam):
            try:
                if model.camera(i).name == name: return i
            except Exception: pass
    raise ValueError(f"camera introuvable: {name}")


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


def cam_setup(env):
    m = env.env.sim.model
    ids = {c: _cam_id(m, c) for c in IMAGE_KEYS}
    orig = {c: (m.cam_pos[i].copy(), m.cam_quat[i].copy()) for c, i in ids.items()}
    return ids, orig


def perturb(env, ids, orig, trans_m, rot_rad, rng):
    m = env.env.sim.model
    for cam, i in ids.items():
        p0, q0 = orig[cam]
        if trans_m > 0:
            d = rng.normal(size=3); d = d / (np.linalg.norm(d) + 1e-9) * rng.uniform(0, trans_m)
        else:
            d = np.zeros(3)
        m.cam_pos[i] = p0 + d
        if rot_rad > 0:
            ax = rng.normal(size=3); ang = rng.uniform(0, rot_rad)
            m.cam_quat[i] = _quat_mult(_axis_angle_quat(ax, ang), q0)
        else:
            m.cam_quat[i] = q0
    env.env.sim.forward()


def eval_level(ckpt, states, device, trans_m, rot_rad, base_seed, max_steps):
    policy, pre, post = load(ckpt, device)
    succ = []; t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = can_eval.make_env(); ids, orig = cam_setup(env)
        for ep in range(i, min(i + CHUNK, len(states))):
            obs = env.reset_to({"states": states[ep]}); policy.reset()
            perturb(env, ids, orig, trans_m, rot_rad, np.random.default_rng(base_seed + ep))
            ok = False
            for _ in range(max_steps):
                images = {}
                for cam, key in IMAGE_KEYS.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                st = torch.from_numpy(state_proprio(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                with torch.no_grad():
                    a = policy.select_action(od)
                a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)  # OSC cartésien
                obs = env.step(a)[0]
                if env.is_success()["task"]: ok = True; break
            succ.append(ok)
        try: env.env.close()
        except Exception: pass
    k = sum(succ); n = len(succ); lo, hi = wilson_ci(k, n)
    return {"n_success": k, "n": n, "success_rate": k / n, "ci95_low": lo, "ci95_high": hi,
            "elapsed_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", required=True, help="trans_cm:rot_deg séparés par virgules")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)[: a.n]
    levels = [tuple(float(x) for x in lv.split(":")) for lv in a.levels.split(",")]
    out = Path(a.out); done = set()
    if out.exists():
        for r in csv.DictReader(open(out)): done.add((float(r["trans_cm"]), float(r["rot_deg"])))
    print(f"[09] ckpt={a.ckpt} device={device} n={a.n} levels={levels}", flush=True)
    wh = not out.exists()
    for trans_cm, rot_deg in levels:
        if (trans_cm, rot_deg) in done:
            print(f"[09] {trans_cm}cm/{rot_deg}deg déjà fait, skip", flush=True); continue
        res = eval_level(a.ckpt, states, device, trans_cm / 100.0, math.radians(rot_deg), a.seed, a.max_steps)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trans_cm", "rot_deg", *res.keys()])
            if wh: w.writeheader(); wh = False
            w.writerow({"trans_cm": trans_cm, "rot_deg": rot_deg, **res})
        print(f"[09] {trans_cm}cm/{rot_deg}deg : {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
              f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
    print("[09] DONE", flush=True)


if __name__ == "__main__":
    main()
