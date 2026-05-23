"""Phase 4 — succès vs nombre de pas de diffusion (le levier latence gratuit).

Rollout du modèle gagnant [32,64,128] sur les 50 init states val figés, en faisant varier
num_inference_steps. Dit jusqu'où on peut réduire les pas (donc la latence) en gardant 100 %.
À croiser avec 56_bench_latency (latence par nb de pas).

Sortie : results/runs/lift/51_unet_sweep_eval/inference_steps.{json,png}
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src import lift_eval

CKPT = "results/runs/lift/51_unet_d32_64_128/checkpoints/007500/pretrained_model"
OUT = Path("results/runs/lift/51_unet_sweep_eval")
STEPS_LIST = [10, 5, 4, 2, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps-list", type=int, nargs="+", default=STEPS_LIST)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    policy = DiffusionPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(args.checkpoint, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(args.checkpoint, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    results = []
    for steps in args.steps_list:
        print(f"\n##### num_inference_steps = {steps} #####", flush=True)
        _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
            device=device, num_inference_steps=steps)
        rec = {"inference_steps": steps, **agg}
        results.append(rec)
        print(f"  => steps={steps}: success={agg['success_rate']:.0%} "
              f"t_succ={agg['t_success_median']} max_z={agg['max_z_mean']:.3f}", flush=True)
        (OUT / "inference_steps.json").write_text(json.dumps(
            {"checkpoint": args.checkpoint, "n": len(val_idx), "results": results}, indent=2))

    # plot succès vs pas
    xs = [r["inference_steps"] for r in results]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, [r["success_rate"] * 100 for r in results], "o-", color="#d62728", label="succès")
    ax1.set_xlabel("num_inference_steps (pas de diffusion)"); ax1.set_ylabel("succès (%)", color="#d62728")
    ax1.set_ylim(-5, 105); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["t_success_median"] if r["t_success_median"] is not None else float("nan") for r in results],
             "s--", color="#1f77b4", label="t_success")
    ax2.set_ylabel("t_success médian", color="#1f77b4")
    fig.suptitle("[32,64,128] — succès vs pas de diffusion (50 val ep)")
    fig.tight_layout(); fig.savefig(OUT / "inference_steps.png", dpi=110)
    print(f"\nJSON: {OUT/'inference_steps.json'}\nPlot: {OUT/'inference_steps.png'}")


if __name__ == "__main__":
    main()
