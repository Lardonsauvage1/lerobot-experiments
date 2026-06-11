"""Phase 5 — Éval 500-rollouts des checkpoints du CONTINUE (prolonge la courbe 30k->).

Évalue les checkpoints de 26_resnet34_dense_continue (steps 1000, 2000, ... dispo),
mêmes 500 départs figés (can_eval500.npy) que le 03. Écrit le succès en STEP ABSOLU
= 30000 + step_continue, pour s'aligner avec rollouts_500.csv (0-30k).

Tourne sur Mac (MPS). Lancer (après seed2) :
  nohup venv312/bin/python -u experiments/phase5_methodology/10_eval_continue_500.py \
    > results/logs/phase5_methodology/run_10_continue500.log 2>&1 &
"""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv, math, time
from pathlib import Path
import numpy as np
import torch

from src import can_eval

CONT_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense_continue")
CKPT_ROOT = CONT_DIR / "checkpoints"
OUT_CSV = CONT_DIR / "rollouts_500_continue.csv"
STATES_PATH = Path("results/runs/can/can_eval500.npy")
BASE_STEP = 30000          # le continue part des poids 30k -> step absolu = 30000 + step_continue
STEPS_INFER = 10
EVAL_INTERVAL = 1000
MAX_CONT_STEP = 10000      # on s'arrête à 40k absolu pour l'instant (les 10 premiers)


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
    per_all = []
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
            per_all.append((t_success is not None, t_success))
        try:
            env.env.close()
        except Exception:
            pass
    return per_all


def eval_ckpt(ckpt_dir, states, device, image_keys):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    import json as _json
    ckpt_path = Path(ckpt_dir) / "pretrained_model"
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = ckpt_path / fn
        if not p.exists():
            continue
        cfg = _json.loads(p.read_text()); changed = False
        for st in cfg.get("steps", []):
            if st.get("registry_name") == "device_processor" and st.get("config", {}).get("device") != device.type:
                st["config"]["device"] = device.type; changed = True
        if changed:
            p.write_text(_json.dumps(cfg, indent=2))
    ckpt = str(ckpt_path)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS_INFER
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    t0 = time.time()
    per = rollout_can(policy, pre, post, states, device, image_keys)
    elapsed = time.time() - t0
    k = sum(s for s, _ in per); n = len(per)
    lo, hi = wilson_ci(k, n)
    ts = [t for s, t in per if s]
    tmed = float(np.median(ts)) if ts else None
    del policy
    return {"n_success": k, "n": n, "success_rate": k / n, "ci95_low": lo, "ci95_high": hi,
            "t_success_median": tmed, "elapsed_sec": elapsed}


def done_steps():
    d = set()
    if OUT_CSV.exists():
        for r in csv.DictReader(open(OUT_CSV)):
            d.add(int(r["step_abs"]))
    return d


def append_csv(row):
    wh = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if wh:
            w.writeheader()
        w.writerow(row)


def stable_ckpt(p):
    """checkpoint prêt = model.safetensors présent, taille > 80 Mo et stable 6 s
    (évite d'évaluer un fichier en cours de transfert)."""
    f = p / "pretrained_model" / "model.safetensors"
    if not f.exists():
        return False
    s1 = f.stat().st_size
    if s1 < 80 * 1024 * 1024:
        return False
    time.sleep(6)
    return f.stat().st_size == s1


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)
    image_keys = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
    print(f"[10] device={device}, {states.shape[0]} états figés — mode poll", flush=True)
    all_targets = [BASE_STEP + cs for cs in range(EVAL_INTERVAL, MAX_CONT_STEP + 1, EVAL_INTERVAL)]
    while True:
        done = done_steps()
        if all(sa in done for sa in all_targets):
            break
        progressed = False
        for p in sorted(CKPT_ROOT.iterdir()):
            if not (p.is_dir() and p.name.isdigit()):
                continue
            cs = int(p.name)
            if cs % EVAL_INTERVAL != 0 or not (0 < cs <= MAX_CONT_STEP):
                continue
            sa = BASE_STEP + cs
            if sa in done or not stable_ckpt(p):
                continue
            try:
                res = eval_ckpt(p, states, device, image_keys)
                append_csv({"step_abs": sa, "step_continue": cs, **res})
                print(f"[10] step_abs={sa} (cont {cs}) : {res['n_success']}/{res['n']} = "
                      f"{res['success_rate']:.1%} [{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] "
                      f"({res['elapsed_sec']:.0f}s)", flush=True)
                progressed = True
            except Exception as e:
                print(f"[10] ⚠ erreur abs={sa} (peut-être transfert incomplet) : {e}", flush=True)
        if not progressed:
            print("[10] rien de nouveau prêt, attente 60s...", flush=True)
            time.sleep(60)
    print("[10] EVAL CONTINUE 30-40k DONE", flush=True)


if __name__ == "__main__":
    main()
