#!/usr/bin/env python
"""Évalue UN checkpoint de la courbe de budget et l'ajoute au JSON commun.

Sert à traiter les points de la courbe au fil de l'eau sur le Mac pendant qu'atomman
continue d'entraîner, au lieu d'attendre la fin pour tout enchaîner.

Usage : venv312/bin/python experiments/can/137_budget_eval.py --ckpt <dir> --tag 025000
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis500)

import torch

OUT = Path("results/runs/can/budget_curve.json")
KEYS = {"agentview": "observation.images.agentview",
        "robot0_eye_in_hand": "observation.images.wrist"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    if not Path(a.ckpt, "model.safetensors").exists():
        print(f"checkpoint absent : {a.ckpt}"); return 1
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.tag in done:
        print(f"{a.tag} déjà évalué : {done[a.tag]['success_rate']:.1%}"); return 0

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # checkpoint entraîné sur XPU -> réécrire le device dans les JSON de pipeline
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(a.ckpt) / f
        if q.exists():
            t = q.read_text()
            if '"xpu"' in t or '"cuda"' in t:
                q.write_text(t.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    policy = DiffusionPolicy.from_pretrained(a.ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = vis500.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = vis500.make_eval_states()[:a.n]
    per, elapsed = vis500.rollout_can(policy, pre, post, states, device, KEYS)
    k = sum(e["success"] for e in per)
    lo, hi = vis500.wilson_ci(k, len(per))
    done[a.tag] = {"ckpt": a.ckpt, "n": len(per), "n_success": k,
                   "success_rate": k / len(per), "ci95": [lo, hi],
                   "elapsed_min": elapsed / 60,
                   "per_episode": [{"ep": i, "success": bool(e["success"])} for i, e in enumerate(per)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(done, indent=2))
    print(f"\n=> budget {a.tag} : {k}/{len(per)} = {k/len(per):.1%} [{lo:.1%}, {hi:.1%}]", flush=True)

    print("\n=== COURBE DE BUDGET (côté + poignet, LR constant) ===")
    print(f"  {'référence 10k (cosine, 3 répliques)':38s} 54,8 % ± 6,5")
    for t in sorted(k2 for k2 in done if k2 != "meta"):
        r = done[t]
        print(f"  {t + ' steps':38s} {r['success_rate']*100:5.1f} % "
              f"[{r['ci95'][0]*100:.1f}-{r['ci95'][1]*100:.1f}]  (n={r['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
