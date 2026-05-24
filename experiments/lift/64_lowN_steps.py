"""Phase 5 — les pas de diffusion aident-ils à N faible ? (sans réentraînement)

Ré-évalue les modèles N=10 et N=20 (mini-CNN) à 4 ET 10 pas, sur les 50 val figés,
pour voir si plus de pas récupère du succès quand le modèle est limité par les données.

Sortie : results/runs/lift/63_data_efficiency_steps.json
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
from pathlib import Path

import torch

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

# (N démos, checkpoint best) — repris de 63_data_efficiency
MODELS = [
    (10, "results/runs/lift/63_dataeff_n10/checkpoints/003000/pretrained_model"),
    (20, "results/runs/lift/63_dataeff_n20/checkpoints/005250/pretrained_model"),
]
STEPS_LIST = [4, 10]
OUT = Path("results/runs/lift/63_data_efficiency_steps.json")


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = lift_eval.load_or_make_split()["val"]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    results = []
    for N, ckpt in MODELS:
        policy = load_minicnn_policy(ckpt, device)
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        print(f"\n##### N={N} démos #####", flush=True)
        for steps in STEPS_LIST:
            _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
                device=device, num_inference_steps=steps, verbose=False)
            results.append({"n_demos": N, "steps": steps, **agg})
            print(f"  {steps:2d} pas : succès {agg['success_rate']:.0%} "
                  f"t_succ={agg['t_success_median']} max_z={agg['max_z_mean']:.3f}", flush=True)
            OUT.write_text(json.dumps({"results": results}, indent=2))
        del policy, pre, post

    print("\n=== Δ (10 pas vs 4 pas) ===")
    for N, _ in MODELS:
        r4 = next(r for r in results if r["n_demos"] == N and r["steps"] == 4)
        r10 = next(r for r in results if r["n_demos"] == N and r["steps"] == 10)
        print(f"  N={N} : {r4['success_rate']:.0%} (4 pas) → {r10['success_rate']:.0%} (10 pas)  "
              f"[Δ {100*(r10['success_rate']-r4['success_rate']):+.0f} pts]")
    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
