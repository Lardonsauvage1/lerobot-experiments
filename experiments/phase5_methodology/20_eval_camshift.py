"""Éval robustesse au DÉPLACEMENT des caméras (Can, modèle 2-cams deja entraine).

On reutilise un checkpoint existant (agentview + birdview + proprio). A chaque épisode,
on decale ALEATOIREMENT et INDEPENDAMMENT les deux cameras (translation + rotation) dans
la limite d'un "niveau", puis on rend depuis ces poses perturbees. La camera reste fixe
PENDANT l'episode (= camera re-fixee un peu de travers pour ce deploiement), differente a
chaque episode. Sweep de niveaux -> courbe succes vs deplacement.

Usage :
  venv312/bin/python -u experiments/phase5_methodology/20_eval_camshift.py \
    --ckpt results/runs/phase5_methodology/mini_constant_continue/checkpoints/046000/pretrained_model \
    --levels 0:0,2:2,5:5,10:10  --n 100 --workers 6 --max-steps 200 --infer-steps 4 \
    --out results/runs/phase5_methodology/camshift_46k.csv
"""
import argparse, csv, math, os, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np

STATES_PATH = "results/runs/can/can_eval500.npy"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    return ((p + z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d,
            (p + z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d)


def _state_proprio(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-9: return np.array([1.0, 0, 0, 0])
    axis = axis / n; h = angle / 2.0; s = math.sin(h)
    return np.array([math.cos(h), s*axis[0], s*axis[1], s*axis[2]])  # wxyz


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


def _eval_slice(ckpt, idx_list, max_steps, infer_steps, trans_m, rot_rad, base_seed):
    import torch
    from src.quiet_robosuite import silence_robosuite; silence_robosuite()
    from src import can_eval
    from src.mini_cnn import load_minicnn_from_ckpt
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    states = np.load(STATES_PATH)
    policy = load_minicnn_from_ckpt(ckpt, device); policy.diffusion.num_inference_steps = infer_steps
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    env = can_eval.make_env()
    def cam_setup(e):
        m = e.env.sim.model
        ids = {c: _cam_id(m, c) for c in IMAGE_KEYS}
        orig = {c: (m.cam_pos[i].copy(), m.cam_quat[i].copy()) for c, i in ids.items()}
        return ids, orig
    ids, orig = cam_setup(env)

    res = []
    for c, ep in enumerate(idx_list):
        obs = env.reset_to(dict(states=states[ep])); policy.reset(); succ = False
        # --- perturbe les 2 cameras (aleatoire par episode, independant par cam) ---
        rng = np.random.default_rng(base_seed + ep)
        m = env.env.sim.model
        for cam, i in ids.items():
            p0, q0 = orig[cam]
            if trans_m > 0:
                d = rng.normal(size=3); d = d/ (np.linalg.norm(d)+1e-9) * rng.uniform(0, trans_m)
            else:
                d = np.zeros(3)
            m.cam_pos[i] = p0 + d
            if rot_rad > 0:
                ax = rng.normal(size=3); ang = rng.uniform(0, rot_rad)
                m.cam_quat[i] = _quat_mult(_axis_angle_quat(ax, ang), q0)
            else:
                m.cam_quat[i] = q0
        env.env.sim.forward()
        # --- rollout ---
        for step_i in range(max_steps):
            images = {}
            for cam, key in IMAGE_KEYS.items():
                img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                images[key] = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
            st = torch.from_numpy(_state_proprio(obs))
            od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
            with torch.no_grad(): a = policy.select_action(od)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs = env.step(a)[0]
            if env.is_success()["task"]: succ = True; break
        res.append(succ)
        if (c + 1) % 50 == 0:
            try: env.env.close()
            except Exception: pass
            env = can_eval.make_env(); ids, orig = cam_setup(env)
    try: env.env.close()
    except Exception: pass
    return res


def _patch_device_cpu(ckpt):
    import json
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = Path(ckpt) / fn
        if not p.exists(): continue
        c = json.loads(p.read_text()); ch = False
        for st in c.get("steps", []):
            if st.get("registry_name") == "device_processor" and st.get("config", {}).get("device") != "cpu":
                st["config"]["device"] = "cpu"; ch = True
        if ch: p.write_text(json.dumps(c, indent=2))


def eval_level(ckpt, n, k, max_steps, infer_steps, trans_m, rot_rad, base_seed):
    import multiprocessing as mp
    _patch_device_cpu(ckpt)
    idx = list(range(n)); slices = [idx[i::k] for i in range(k)]
    ctx = mp.get_context("spawn"); t0 = time.time()
    with ctx.Pool(k) as pool:
        out = pool.starmap(_eval_slice, [(ckpt, sl, max_steps, infer_steps, trans_m, rot_rad, base_seed) for sl in slices])
    allr = [x for sub in out for x in sub]
    ks = sum(allr); nt = len(allr); lo, hi = wilson_ci(ks, nt)
    return {"n_success": ks, "n": nt, "success_rate": ks/nt, "ci95_low": lo, "ci95_high": hi, "elapsed_sec": time.time()-t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", required=True, help="liste trans_cm:rot_deg separee par virgules, ex 0:0,2:2,5:5,10:10")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    levels = [tuple(float(x) for x in lv.split(":")) for lv in a.levels.split(",")]
    out = Path(a.out)
    print(f"[20] ckpt={a.ckpt}  n={a.n} workers={a.workers} levels={levels}", flush=True)
    wh = not out.exists()
    for trans_cm, rot_deg in levels:
        res = eval_level(a.ckpt, a.n, a.workers, a.max_steps, a.infer_steps,
                         trans_cm/100.0, math.radians(rot_deg), a.seed)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trans_cm","rot_deg",*res.keys()])
            if wh: w.writeheader(); wh = False
            w.writerow({"trans_cm": trans_cm, "rot_deg": rot_deg, **res})
        print(f"[20] trans={trans_cm}cm rot={rot_deg}deg : {res['n_success']}/{res['n']} = "
              f"{res['success_rate']:.1%} [{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
    print("[20] DONE", flush=True)


if __name__ == "__main__":
    main()
