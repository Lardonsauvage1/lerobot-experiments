"""Phase 5 — Éval 500-rollouts FINE (tous les 100 steps) sur le modèle dense.
But : isoler la VRAIE variation du modèle (chaque point fiable ±4 pts) à
résolution 100 steps, vs la variation de mesure.

Paramétrable : --steps 10000,10100,... --out chemin.csv
Tourne sur Mac (MPS) ou atomman (CPU) — device auto. États figés can_eval500.npy.
"""
import sys, argparse
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv, math, time
from pathlib import Path
import numpy as np
import torch
from src import can_eval

CKPT_ROOT = Path("results/runs/phase5_methodology/26_resnet34_dense/checkpoints")
STATES_PATH = Path("results/runs/can/can_eval500.npy")
STEPS_INFER = 10


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n
    return ((p+z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d,
            (p+z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d)


def state_proprio(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def rollout_can(policy, pre, post, states, device, image_keys, chunk=50):
    per = []
    for i in range(0, len(states), chunk):
        env = can_eval.make_env()
        for ep in range(i, min(i+chunk, len(states))):
            obs = env.reset_to(dict(states=states[ep])); policy.reset(); ts = None
            for step_i in range(can_eval.MAX_STEPS):
                images = {}
                for cam, key in image_keys.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
                st = torch.from_numpy(state_proprio(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                with torch.no_grad(): a = policy.select_action(od)
                a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
                obs = env.step(a)[0]
                if env.is_success()["task"]: ts = step_i; break
            per.append(ts is not None)
        try: env.env.close()
        except Exception: pass
    return per


def eval_ckpt(p, states, device, image_keys):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    import json as _json
    cp = Path(p) / "pretrained_model"
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        f = cp/fn
        if not f.exists(): continue
        c = _json.loads(f.read_text()); ch = False
        for st in c.get("steps", []):
            if st.get("registry_name")=="device_processor" and st.get("config",{}).get("device")!=device.type:
                st["config"]["device"]=device.type; ch=True
        if ch: f.write_text(_json.dumps(c, indent=2))
    ckpt = str(cp)
    pol = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    pol.diffusion.num_inference_steps = STEPS_INFER
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    t0 = time.time()
    per = rollout_can(pol, pre, post, states, device, image_keys); el = time.time()-t0
    k = sum(per); n = len(per); lo, hi = wilson_ci(k, n); del pol
    return {"n_success": k, "n": n, "success_rate": k/n, "ci95_low": lo, "ci95_high": hi, "elapsed_sec": el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    out = Path(args.out)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    print(f"[11] device={device}, {len(steps)} steps, out={out.name}", flush=True)
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)): done.add(int(r["step"]))
    for s in steps:
        if s in done: continue
        p = CKPT_ROOT / f"{s:06d}"
        if not (p/"pretrained_model"/"model.safetensors").exists():
            print(f"[11] step {s}: checkpoint absent, skip", flush=True); continue
        try:
            res = eval_ckpt(p, states, device, image_keys)
            wh = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", *res.keys()])
                if wh: w.writeheader()
                w.writerow({"step": s, **res})
            print(f"[11] step={s}: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
        except Exception as e:
            print(f"[11] erreur step={s}: {e}", flush=True)
    print("[11] DONE", flush=True)


if __name__ == "__main__":
    main()
