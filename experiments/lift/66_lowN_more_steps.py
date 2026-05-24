"""Phase 5 — N=10 : jusqu'où les pas de diffusion compensent-ils le manque de données ?

Ré-évalue le modèle N=10 (mini-CNN) à 4 / 10 / 20 / 50 pas (sans réentraîner), succès + latence
sur les 50 val figés. Le succès grimpe-t-il vers 100 % avec plus de pas, ou plafonne-t-il ?

Sortie : results/runs/lift/63_dataeff_n10/more_steps.{json,png}
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

CKPT = "results/runs/lift/63_dataeff_n10/checkpoints/003000/pretrained_model"
OUT = Path("results/runs/lift/63_dataeff_n10/more_steps")
STEPS_LIST = [4, 10, 20, 50]


def fake_obs(device):
    return {"observation.image": torch.rand(1, 3, 96, 96, device=device),
            "observation.state": torch.rand(1, 19, device=device)}


def measure_latency(policy, device, steps, n=6):
    policy.diffusion.num_inference_steps = steps
    sync = (lambda: torch.mps.synchronize()) if device.type == "mps" else (lambda: None)
    with torch.no_grad():
        for _ in range(2):
            policy.reset(); policy.select_action(fake_obs(device)); sync()
        ts = []
        for _ in range(n):
            policy.reset(); sync(); t0 = time.perf_counter()
            policy.select_action(fake_obs(device)); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


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
    policy = load_minicnn_policy(CKPT, device)
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    results = []
    for steps in STEPS_LIST:
        lat = measure_latency(policy, device, steps)
        _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
            device=device, num_inference_steps=steps, verbose=False)
        results.append({"steps": steps, "latency_ms": lat, **agg})
        print(f"  {steps:2d} pas : {lat:6.1f} ms · succès {agg['success_rate']:.0%} "
              f"t_succ={agg['t_success_median']} max_z={agg['max_z_mean']:.3f}", flush=True)
        OUT.with_suffix(".json").write_text(json.dumps({"checkpoint": CKPT, "results": results}, indent=2))

    xs = [r["steps"] for r in results]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, [r["success_rate"] * 100 for r in results], "o-", color="#d62728", label="succès")
    for r in results:
        ax1.annotate(f"{r['success_rate']:.0%}", (r["steps"], r["success_rate"] * 100),
                     textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.set_xlabel("pas de diffusion"); ax1.set_ylabel("succès (%)", color="#d62728"); ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["latency_ms"] for r in results], "s--", color="#1f77b4", label="latence")
    ax2.set_ylabel("latence (ms)", color="#1f77b4")
    fig.suptitle("N=10 démos — succès & latence vs pas de diffusion")
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
