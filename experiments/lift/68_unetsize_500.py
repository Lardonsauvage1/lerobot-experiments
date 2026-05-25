"""Phase 4 — taille de U-Net, éval RIGOUREUSE à 500 rollouts (pendant de 67 pour les données).

Le sweep U-Net donnait [128,256,512]=98 %, [64,128,256]=100 %, [32,64,128]=100 % à 50 ép. :
ces écarts sont DANS LE BRUIT (±~7 pts). On re-mesure le succès sur les MÊMES 500 départs
figés que 67 (phase4_eval500.npy) → comparaison appariée, IC95 ≈ ±2 pts.

Modèles = sweep ResNet18, 150 démos, best checkpoint (val-loss) identique à 52_sweep_eval :
  - [128,256,512] (28.7 M) / 004500
  - [64,128,256]  (16.2 M) / 006000
  - [32,64,128]   (12.8 M) / 007500   <- ancre ResNet18 (vision fixée pour isoler la taille)
Tous @ 4 pas (point de fonctionnement), arrêt au 1er succès.

NB : vision = ResNet18 ici (sweep), alors que 67 N=150 utilise le mini-CNN [32,64,128].
La comparaison de TAILLE est à vision constante (ResNet18) au sein de ce script.

Sortie : results/runs/lift/68_unetsize_500.{json,png}  (réutilise phase4_eval500.npy)
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import lift_eval

STEPS = 4
STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
OUT = Path("results/runs/lift/68_unetsize_500")
# (label, params M, checkpoint best val-loss — IDENTIQUE à 52_sweep_eval.json)
RUNS = [
    ("[128,256,512]", 28.7, "results/runs/lift/51_unet_d128_256_512/checkpoints/004500/pretrained_model"),
    ("[64,128,256]",  16.2, "results/runs/lift/51_unet_d64_128_256/checkpoints/006000/pretrained_model"),
    ("[32,64,128]",   12.8, "results/runs/lift/51_unet_d32_64_128/checkpoints/007500/pretrained_model"),
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
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if not STATES_PATH.exists():
        raise SystemExit(f"{STATES_PATH} absent — lance d'abord 67_dataeff_500.py (génère les 500 états).")
    states = np.load(STATES_PATH)
    print(f"États de test (partagés avec 67) : {states.shape}", flush=True)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def load(ckpt):
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return policy, pre, post

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for label, params_m, ckpt in RUNS:
        print(f"\n##### U-Net {label} ({params_m} M) — {ckpt} #####", flush=True)
        policy, pre, post = load(ckpt)
        per_ep, agg = lift_eval.rollout_eval(policy, pre, post, env, states,
            device=device, num_inference_steps=STEPS, verbose=False, stop_on_success=True)
        del policy
        k = sum(e["success"] for e in per_ep)
        lo, hi = wilson_ci(k, len(per_ep))
        rec = {"down_dims": label, "params_m": params_m, "checkpoint": ckpt, "steps": STEPS,
               "n": len(per_ep), "n_success": k, "success_rate": agg["success_rate"],
               "ci95_low": lo, "ci95_high": hi,
               "t_success_median": agg["t_success_median"], "elapsed_s": agg["elapsed_s"]}
        results.append(rec)
        print(f"  => {label} : {k}/{len(per_ep)} = {agg['success_rate']:.1%} "
              f"[{lo:.1%}, {hi:.1%}]  t_succ_med={agg['t_success_median']} "
              f"({agg['elapsed_s'] / 60:.1f} min)", flush=True)
        OUT.with_suffix(".json").write_text(json.dumps({"steps": STEPS, "results": results}, indent=2))

    # plot succès vs params avec barres d'erreur Wilson
    results.sort(key=lambda r: r["params_m"])
    xs = [r["params_m"] for r in results]
    ys = [r["success_rate"] * 100 for r in results]
    err_lo = [(r["success_rate"] - r["ci95_low"]) * 100 for r in results]
    err_hi = [(r["ci95_high"] - r["success_rate"]) * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o-", color="#1f77b4", capsize=4)
    for r in results:
        ax.annotate(f"{r['down_dims']}\n{r['success_rate']:.0%}", (r["params_m"], r["success_rate"] * 100),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
    ax.set_xlabel("params total (M, ResNet18 + U-Net)")
    ax.set_ylabel(f"succès (%) @ {STEPS} pas, 500 départs (IC95 Wilson)")
    ax.set_ylim(80, 103); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    ax.set_title("Taille de U-Net vs succès — 500 rollouts appariés (vision ResNet18)")
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
