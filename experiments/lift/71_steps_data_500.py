"""Phase 4 — surface PAS × DONNÉES, 500 rollouts (modèle [32,64,128] mini-CNN).

Vérifie rigoureusement (500 rollouts, IC95 Wilson) deux affirmations issues de 50 ép. :
  - "plancher = 4 pas" (2 casse ? 4 suffit ? plus aide ?) — à pleines données ET à N faible.
  - "monter les pas récupère du succès à N faible" (le 'pont' pas↔données).

Bouton d'INFÉRENCE : aucun réentraînement, on varie num_inference_steps sur les checkpoints
[32,64,128] existants (mêmes que 67/63), mêmes 500 départs figés, env recréé par tranche.
Colonne @4 pas = déjà dans 67 → ici on fait pas {2,10,20,50}.

Sortie : results/runs/lift/71_steps_data_500.json  (incrémental, clé = (N, steps))
  --smoke : 1 cellule, 20 départs (pré-vol mécanique).
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
OUT = Path("results/runs/lift/71_steps_data_500.json")
STEPS_LIST = [2, 10, 20, 50]  # 4 pas déjà dans 67
# (N, checkpoint) — IDENTIQUE à 67_dataeff_500 (best val-loss [32,64,128])
RUNS_N = [
    (150, "results/runs/lift/61_minicnn/checkpoints/010500/pretrained_model"),
    (100, "results/runs/lift/63_dataeff_n100/checkpoints/005250/pretrained_model"),
    (50,  "results/runs/lift/63_dataeff_n50/checkpoints/006000/pretrained_model"),
    (20,  "results/runs/lift/63_dataeff_n20/checkpoints/005250/pretrained_model"),
    (10,  "results/runs/lift/63_dataeff_n10/checkpoints/003000/pretrained_model"),
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = np.load(STATES_PATH)
    runs_n = RUNS_N[:1] if args.smoke else RUNS_N
    steps_list = [10] if args.smoke else STEPS_LIST
    if args.smoke:
        states = states[:20]
    print(f"États {states.shape} | N={[n for n,_ in runs_n]} | pas={steps_list}", flush=True)

    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def procs(ckpt):
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return pre, post

    results = []
    if OUT.exists() and not args.smoke:
        results = json.loads(OUT.read_text()).get("results", [])
    done = {(r["n_demos"], r["steps"]) for r in results}

    for N, ckpt in runs_n:
        if all((N, s) in done for s in steps_list):
            print(f"N={N} déjà fait, skip", flush=True); continue
        policy = load_minicnn_policy(ckpt, device)  # une fois par N, on varie les pas
        pre, post = procs(ckpt)
        for steps in steps_list:
            if (N, steps) in done:
                continue
            print(f"\n##### N={N} pas={steps} #####", flush=True)
            per_ep, agg = lift_eval.rollout_eval_chunked(policy, pre, post, states,
                device=device, num_inference_steps=steps, stop_on_success=True)
            k = sum(e["success"] for e in per_ep)
            lo, hi = wilson_ci(k, len(per_ep))
            rec = {"n_demos": N, "steps": steps, "checkpoint": ckpt, "n": len(per_ep),
                   "n_success": k, "success_rate": agg["success_rate"], "ci95_low": lo,
                   "ci95_high": hi, "t_success_median": agg["t_success_median"],
                   "elapsed_s": agg["elapsed_s"]}
            results.append(rec)
            print(f"  => N={N} pas={steps} : {k}/{len(per_ep)} = {agg['success_rate']:.1%} "
                  f"[{lo:.1%}, {hi:.1%}]  t_succ_med={agg['t_success_median']} "
                  f"({agg['elapsed_s'] / 60:.1f} min)", flush=True)
            if not args.smoke:
                OUT.write_text(json.dumps({"model": "[32,64,128]", "n_eval": int(states.shape[0]),
                                           "results": results}, indent=2))
        del policy

    print(f"\n{'SMOKE OK' if args.smoke else 'STEPS-DATA DONE'} — {OUT if not args.smoke else '(smoke, non sauvé)'}")


if __name__ == "__main__":
    main()
