"""Phase 4 / Phase 0 — éval propre d'un checkpoint Diffusion sur le VAL set figé.

Socle réutilisable pour TOUS les modèles de la phase compression (baseline, rétrécis,
quantifiés). Métriques continues (succès + temps-au-succès + marge de levage + maintien)
sur les 50 init states du val set jamais entraînés.

Exemples :
  # smoke test sur le modèle run-46 (entraîné sur 200 démos → val "vue", juste pour valider le harnais)
  python -u experiments/lift/47_phase0_eval.py \
      --checkpoint results/runs/lift/46_diffusion_official/checkpoints/012000/pretrained_model

  # même chose, 4 pas de diffusion, sur 10 épisodes (rapide)
  python -u experiments/lift/47_phase0_eval.py --checkpoint <ckpt> --inference-steps 4 --limit 10
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import lift_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dossier pretrained_model d'un checkpoint")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--inference-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="n'évalue que les N premiers épisodes")
    ap.add_argument("--out", default=None, help="chemin json de sortie (défaut: à côté du checkpoint)")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    split = lift_eval.load_or_make_split()
    indices = split[args.split]
    if args.limit:
        indices = indices[:args.limit]
    print(f"Split: train {len(split['train'])} / val {len(split['val'])} ({split['layout']})")
    print(f"Éval sur {len(indices)} épisodes du set '{args.split}' | inference_steps={args.inference_steps}\n", flush=True)

    init_states = lift_eval.load_init_states(indices)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    ckpt = args.checkpoint
    print(f"Chargement: {ckpt}", flush=True)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    per_ep, agg = lift_eval.rollout_eval(
        policy, pre, post, env, init_states,
        device=device, max_steps=args.max_steps, num_inference_steps=args.inference_steps,
    )

    print("\n=== AGRÉGAT ===")
    print(f"  success_rate      : {agg['success_rate']:.0%}  ({agg['n']} ép.)")
    tsm = agg["t_success_median"]
    print(f"  t_success médian  : {tsm:.0f} steps" if tsm is not None else "  t_success médian  : —")
    print(f"  max_z moyen       : {agg['max_z_mean']:.3f}")
    print(f"  hold_fraction moy : {agg['hold_fraction_mean']:.2f}")
    print(f"  durée             : {agg['elapsed_s']/60:.1f} min")

    out = Path(args.out) if args.out else Path(ckpt).parent.parent / "phase0_eval.json"
    out.write_text(json.dumps({
        "checkpoint": ckpt, "split": args.split, "indices": list(map(int, indices)),
        "inference_steps": args.inference_steps, "max_steps": args.max_steps,
        "aggregate": agg, "per_episode": per_ep,
    }, indent=2))
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
