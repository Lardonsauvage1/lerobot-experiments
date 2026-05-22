"""Phase 4 — latence d'inférence : le vrai objectif.

Mesure le temps d'une décision pour les modèles clés, en faisant varier le nombre de pas
de diffusion (levier gratuit, sans réentraînement). Une décision « pleine » = 1 encodage vision
+ num_inference_steps passes du U-Net. Le 'queue pop' = action déjà dans le buffer (quasi gratuit).

Latence amortie par step d'env = (1 sample + (n_action_steps-1) pops) / n_action_steps.

Sortie : results/runs/lift/51_unet_sweep_eval/latency.{json,png}
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

OUT = Path("results/runs/lift/51_unet_sweep_eval")
N_WARMUP = 3
N_TRIALS = 8
STEPS_LIST = [10, 5, 4, 2, 1]
MODELS = [
    ("baseline 263.7M", "results/runs/lift/47_baseline_150/checkpoints/006000/pretrained_model"),
    ("[64,128,256] 16.2M", "results/runs/lift/51_unet_d64_128_256/checkpoints/006000/pretrained_model"),
    ("[32,64,128] 12.8M", "results/runs/lift/51_unet_d32_64_128/checkpoints/007500/pretrained_model"),
]


def fake_obs(device):
    return {"observation.image": torch.rand(1, 3, 96, 96, device=device),
            "observation.state": torch.rand(1, 19, device=device)}


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    results = []
    for label, ckpt in MODELS:
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        n_action = policy.config.n_action_steps
        params = sum(p.numel() for p in policy.parameters())
        print(f"\n##### {label} ({params/1e6:.1f}M, n_action_steps={n_action}) #####", flush=True)

        # queue pop (indépendant du nb de pas) : on mesure une fois
        policy.diffusion.num_inference_steps = 10
        with torch.no_grad():
            policy.reset(); _ = policy.select_action(fake_obs(device))  # amorce la queue
            pops = []
            for _ in range(N_TRIALS):
                sync(device); t0 = time.perf_counter()
                _ = policy.select_action(fake_obs(device))
                sync(device); pops.append(time.perf_counter() - t0)
        pop_ms = float(np.median(pops)) * 1000

        for steps in STEPS_LIST:
            policy.diffusion.num_inference_steps = steps
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    policy.reset(); _ = policy.select_action(fake_obs(device)); sync(device)
                samp = []
                for _ in range(N_TRIALS):
                    policy.reset()
                    sync(device); t0 = time.perf_counter()
                    _ = policy.select_action(fake_obs(device))
                    sync(device); samp.append(time.perf_counter() - t0)
            sample_ms = float(np.median(samp)) * 1000
            amort_ms = (sample_ms + (n_action - 1) * pop_ms) / n_action  # latence moyenne par step d'env
            results.append({"model": label, "params": params, "inference_steps": steps,
                            "sample_ms": sample_ms, "pop_ms": pop_ms, "amortized_ms_per_envstep": amort_ms})
            print(f"  steps={steps:2d} : sample={sample_ms:6.1f} ms | pop={pop_ms:4.1f} ms | "
                  f"amorti/env-step={amort_ms:6.1f} ms", flush=True)
        del policy

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps({"device": device.type, "results": results}, indent=2))

    # plot : sample_ms vs inference_steps, une courbe par modèle
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, _ in MODELS:
        rs = [r for r in results if r["model"] == label]
        ax.plot([r["inference_steps"] for r in rs], [r["sample_ms"] for r in rs], "o-", label=label)
    ax.set_xlabel("num_inference_steps (pas de diffusion)"); ax.set_ylabel("latence d'un sample (ms)")
    ax.set_title(f"Latence diffusion vs pas — device {device.type}"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "latency.png", dpi=110)
    print(f"\nJSON: {OUT/'latency.json'}\nPlot: {OUT/'latency.png'}")


if __name__ == "__main__":
    main()
