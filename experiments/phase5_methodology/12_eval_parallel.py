"""Éval rollouts PARALLÈLE (lever 1) + MAX_STEPS réduit (lever 4) — pour modèles mini-CNN.

Découpe les N départs figés (can_eval500.npy) en K tranches traitées en PARALLÈLE
(K processus CPU), puis fusionne → succès + IC95 Wilson. ~K× plus rapide sur la sim.

Usage :
  venv312/bin/python -u experiments/phase5_methodology/12_eval_parallel.py \
    --run-dir results/runs/phase5_methodology/mini_cosine \
    --steps 1000,2000,...   --n 500 --workers 10 --max-steps 200 --infer-steps 4 \
    --out .../rollouts_500.csv
"""
import argparse, csv, math, os, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())   # pour que les workers spawn retrouvent `src`
import numpy as np

STATES_PATH = "results/runs/can/can_eval500.npy"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    return ((p + z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d,
            (p + z*z/(2*n))/d + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d)


def _state_proprio(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def _eval_slice(ckpt, idx_list, max_steps, infer_steps):
    """Tourne dans un processus worker : évalue les épisodes idx_list."""
    import torch
    from src.quiet_robosuite import silence_robosuite; silence_robosuite()
    from src import can_eval
    from src.mini_cnn import load_minicnn_from_ckpt
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    torch.set_num_threads(1)  # chaque worker mono-thread (le parallélisme est entre processus)
    device = torch.device("cpu")
    states = np.load(STATES_PATH)
    policy = load_minicnn_from_ckpt(ckpt, device); policy.diffusion.num_inference_steps = infer_steps
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    res = []
    env = can_eval.make_env()
    for c, ep in enumerate(idx_list):
        obs = env.reset_to(dict(states=states[ep])); policy.reset(); succ = False
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
        if (c + 1) % 50 == 0:  # recrée l'env périodiquement (anti-dégradation renderer)
            try: env.env.close()
            except Exception: pass
            env = can_eval.make_env()
    try: env.env.close()
    except Exception: pass
    return res


def _patch_device_cpu(ckpt):
    """Patche les JSON preprocessor/postprocessor sur cpu (1 seule fois, avant les workers)."""
    import json
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = Path(ckpt) / fn
        if not p.exists(): continue
        c = json.loads(p.read_text()); ch = False
        for st in c.get("steps", []):
            if st.get("registry_name") == "device_processor" and st.get("config", {}).get("device") != "cpu":
                st["config"]["device"] = "cpu"; ch = True
        if ch: p.write_text(json.dumps(c, indent=2))


def eval_ckpt_parallel(ckpt, n, k, max_steps, infer_steps):
    import multiprocessing as mp
    _patch_device_cpu(ckpt)
    idx = list(range(n))
    slices = [idx[i::k] for i in range(k)]  # round-robin (équilibre les durées)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(k) as pool:
        out = pool.starmap(_eval_slice, [(ckpt, sl, max_steps, infer_steps) for sl in slices])
    allr = [x for sub in out for x in sub]
    ks = sum(allr); nt = len(allr); lo, hi = wilson_ci(ks, nt)
    return {"n_success": ks, "n": nt, "success_rate": ks/nt, "ci95_low": lo, "ci95_high": hi,
            "elapsed_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", required=True, help="liste séparée par virgules")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]
    out = Path(a.out); ckroot = Path(a.run_dir) / "checkpoints"
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)): done.add(int(r["step"]))
    print(f"[12] {len(steps)} ckpts, n={a.n}, workers={a.workers}, max_steps={a.max_steps}, infer={a.infer_steps}", flush=True)
    for s in steps:
        if s in done: continue
        ck = ckroot / f"{s:06d}" / "pretrained_model"
        if not (ck / "model.safetensors").exists():
            print(f"[12] step {s}: absent, skip", flush=True); continue
        try:
            res = eval_ckpt_parallel(str(ck), a.n, a.workers, a.max_steps, a.infer_steps)
            wh = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", *res.keys()])
                if wh: w.writeheader()
                w.writerow({"step": s, **res})
            print(f"[12] step={s}: {res['n_success']}/{res['n']} = {res['success_rate']:.1%} "
                  f"[{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] ({res['elapsed_sec']:.0f}s)", flush=True)
        except Exception as e:
            print(f"[12] erreur step={s}: {e}", flush=True)
    print("[12] DONE", flush=True)


if __name__ == "__main__":
    main()
