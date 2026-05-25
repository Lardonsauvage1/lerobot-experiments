"""Phase 4 — efficacité données, éval RIGOUREUSE à 500 rollouts.

Le succès sur 50 ép. a un IC95 ≈ ±7 pts → les écarts entre N (94/96/98/100 %) sont
DANS LE BRUIT (N=20 "bat" N=50 = artefact). La val-loss, elle, ne dit pas si le robot
accomplit la tâche (juste s'il colle aux démos). On garde donc le succès comme juge mais
on lui donne 10x plus de mesures : 500 départs → IC95 ≈ ±2 pts, les N deviennent séparables.

- 500 états de test = resets aléatoires de l'env (cube placé au hasard), jamais entraînés,
  même distribution que les démos. Générés une fois, FIGÉS (npy) et APPARIÉS (mêmes 500
  états pour tous les N → enlève la variance inter-départs de la comparaison).
- Arrêt anticipé au 1er succès : succès = "réussi à un moment" → s'arrêter n'change pas le
  taux (seuls les échecs vont au bout des 200 steps), mais ~2.5x plus rapide.
- Mêmes checkpoints (best val-loss) et même point @ 4 pas que 63_data_efficiency.

Sortie : results/runs/lift/67_dataeff_500.{json,png} + phase4_eval500.npy
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
from src.mini_cnn import load_minicnn_policy

STEPS = 4
N_EVAL = 500
STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
OUT = Path("results/runs/lift/67_dataeff_500")
# (N démos, checkpoint best val-loss — IDENTIQUE à 63_data_efficiency.json)
RUNS = [
    (150, "results/runs/lift/61_minicnn/checkpoints/010500/pretrained_model"),
    (100, "results/runs/lift/63_dataeff_n100/checkpoints/005250/pretrained_model"),
    (50,  "results/runs/lift/63_dataeff_n50/checkpoints/006000/pretrained_model"),
    (20,  "results/runs/lift/63_dataeff_n20/checkpoints/005250/pretrained_model"),
    (10,  "results/runs/lift/63_dataeff_n10/checkpoints/003000/pretrained_model"),
]


def wilson_ci(k, n, z=1.96):
    """IC de Wilson (robuste près de p=1, contrairement à l'approx normale)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def make_eval_states(n=N_EVAL, seed=0, path=STATES_PATH):
    """500 états initiaux figés = resets aléatoires de l'env (jamais entraînés).

    IMPORTANT : génère avec un env JETABLE. Des centaines de reset() dégradent le
    renderer/env offscreen mujoco — il ne faut surtout pas réutiliser cet env pour les
    rollouts (bug observé : 500 resets de génération + rollouts même env → 2.2% au lieu
    de 100%). L'env de rollout est créé séparément, FRAIS, dans main().
    """
    if path.exists():
        arr = np.load(path)
        print(f"États de test chargés : {arr.shape} ({path})", flush=True)
        return arr
    print(f"Génération de {n} états de test (env jetable, resets aléatoires, seed={seed})...", flush=True)
    genenv = lift_eval.make_env()
    np.random.seed(seed)
    states = []
    for i in range(n):
        genenv.reset()
        states.append(np.asarray(genenv.get_state()["states"]).copy())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} états", flush=True)
    arr = np.stack(states)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"Sauvé : {path} {arr.shape}", flush=True)
    return arr


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = make_eval_states()  # env de rollout recréé par tranche dans rollout_eval_chunked

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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for N, ckpt in RUNS:
        print(f"\n##### N={N} démos — {ckpt} #####", flush=True)
        policy = load_minicnn_policy(ckpt, device)
        pre, post = procs(ckpt)
        per_ep, agg = lift_eval.rollout_eval_chunked(policy, pre, post, states,
            device=device, num_inference_steps=STEPS, stop_on_success=True)
        del policy
        k = sum(e["success"] for e in per_ep)
        lo, hi = wilson_ci(k, len(per_ep))
        rec = {"n_demos": N, "checkpoint": ckpt, "steps": STEPS, "n": len(per_ep),
               "n_success": k, "success_rate": agg["success_rate"],
               "ci95_low": lo, "ci95_high": hi,
               "t_success_median": agg["t_success_median"], "elapsed_s": agg["elapsed_s"]}
        results.append(rec)
        print(f"  => N={N} : {k}/{len(per_ep)} = {agg['success_rate']:.1%} "
              f"[{lo:.1%}, {hi:.1%}]  t_succ_med={agg['t_success_median']} "
              f"({agg['elapsed_s'] / 60:.1f} min)", flush=True)
        OUT.with_suffix(".json").write_text(json.dumps(
            {"steps": STEPS, "n_eval": N_EVAL, "results": results}, indent=2))

    # plot succès vs N avec barres d'erreur Wilson
    results.sort(key=lambda r: r["n_demos"])
    xs = [r["n_demos"] for r in results]
    ys = [r["success_rate"] * 100 for r in results]
    err_lo = [(r["success_rate"] - r["ci95_low"]) * 100 for r in results]
    err_hi = [(r["ci95_high"] - r["success_rate"]) * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o-", color="#d62728", capsize=4)
    for r in results:
        ax.annotate(f"{r['success_rate']:.0%}", (r["n_demos"], r["success_rate"] * 100),
                    textcoords="offset points", xytext=(0, 12), ha="center")
    ax.set_xlabel("nb de démos d'entraînement")
    ax.set_ylabel(f"succès (%) @ {STEPS} pas, {N_EVAL} départs (IC95 Wilson)")
    ax.set_ylim(40, 103); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Efficacité données — {N_EVAL} rollouts appariés")
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
