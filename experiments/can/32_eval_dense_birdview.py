"""Éval dense par checkpoint d'un run birdview (DiffusionPolicy standard, ResNet18/34).

Produit une trajectoire succès vs step -> nourrit l'étude de convergence (docs/CONVERGENCE.md).

MPS single-process (comme 10_vision_500_rollouts.py) : les gros modèles diffusion sont
catastrophiquement lents en CPU (~4 min/rollout) -> on évalue sur le GPU MPS.
Réglages identiques à l'éval 500 pour rester comparable : MAX_STEPS=300, infer_steps=10,
états figés can_eval500.npy (n premiers), birdview 2-cam. n=50 par défaut (IC95 ±~13 pts).

Usage :
  venv312/bin/python -u experiments/can/32_eval_dense_birdview.py \
    --run-dir results/runs/can/31_proprio_birdview_r34_bigunet \
    --steps "" --n 50 --out results/runs/can/31_proprio_birdview_r34_bigunet/rollouts_50.csv
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
CHUNK = 50  # recrée l'env tous les 50 ép. (anti-dégradation renderer)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    return ((p + z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d,
            (p + z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d)


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


def eval_ckpt(ckpt, states, device, max_steps):
    policy, pre, post = load(ckpt, device)
    succ = []
    t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = can_eval.make_env()
        for ep in range(i, min(i + CHUNK, len(states))):
            obs = env.reset_to(dict(states=states[ep])); policy.reset(); ok = False
            for _ in range(max_steps):
                images = {}
                for cam, key in IMAGE_KEYS.items():
                    img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                st = torch.from_numpy(state_proprio(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                # robustesse : certains checkpoints ont un preprocessor patché en cpu
                # (ancienne version du script) -> on force les entrées sur le device du modèle.
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                with torch.no_grad():
                    a = policy.select_action(od)
                a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
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
    ap.add_argument("--steps", default="", help="liste séparée par virgules ; vide -> tous les ckpts")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)[: a.n]
    ckroot = Path(a.run_dir) / "checkpoints"
    if a.steps.strip():
        steps = [int(s) for s in a.steps.split(",")]
    else:
        steps = sorted(int(p.name) for p in ckroot.glob("[0-9]" * 6))

    out = Path(a.out); done = set()
    if out.exists():
        for r in csv.DictReader(open(out)): done.add(int(r["step"]))
    print(f"[32] device={device}, {len(steps)} ckpts, n={a.n}, max_steps={a.max_steps}, déjà faits={sorted(done)}", flush=True)
    for s in steps:
        if s in done:
            print(f"[32] step {s}: déjà fait, skip", flush=True); continue
        ck = ckroot / f"{s:06d}" / "pretrained_model"
        if not (ck / "model.safetensors").exists():
            print(f"[32] step {s}: absent, skip", flush=True); continue
        try:
            res = eval_ckpt(str(ck), states, device, a.max_steps)
            wh = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", *res.keys()])
                if wh: w.writeheader()
                w.writerow({"step": s, **res})
            print(f"[32] step={s}: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
        except Exception as e:
            print(f"[32] erreur step={s}: {e}", flush=True)
    print("[32] DONE", flush=True)


if __name__ == "__main__":
    main()
