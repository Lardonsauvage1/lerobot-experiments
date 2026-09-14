#!/usr/bin/env python
"""Diagnostic d'EFFONDREMENT DE MODALITÉ — un modèle bi-caméra utilise-t-il vraiment ses deux entrées ?

Le problème qu'il résout. Quand un bras à deux caméras ne fait pas mieux qu'à une, deux causes
sont indiscernables au seul taux de succès : (a) la deuxième caméra n'apporte rien, (b) la fusion
par concaténation naïve l'a étouffée — l'effondrement de modalité décrit par « Factorizing
Diffusion Policies » (arXiv 2509.16830).

Le test. On rejoue les MÊMES 500 états avec le MÊME modèle, en masquant une caméra à la fois
(remise à zéro APRÈS normalisation = image moyenne, signal « absent » neutre).

    masquer une caméra ne change RIEN  -> le modèle ne s'en servait pas (effondrement)
    masquer la fait chuter             -> elle porte réellement de l'information

C'est un diagnostic, pas une mesure de performance : un modèle amputé n'est pas déployable.

Usage : venv312/bin/python experiments/can/135_ablation_modalites.py --run cam2_E_birdview_wrist
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

OUT = Path("results/runs/can/ablation_modalites.json")

KEYS = {
    "cam2_E_birdview_wrist": {"birdview": "observation.images.birdview",
                              "robot0_eye_in_hand": "observation.images.wrist"},
    "cam2_C2_wrist_fixstats": {"agentview": "observation.images.agentview",
                               "robot0_eye_in_hand": "observation.images.wrist"},
    "cam2_B2_fixstats": {"agentview": "observation.images.agentview",
                         "birdview": "observation.images.birdview"},
}


def load(ckpt, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(ckpt) / f
        if q.exists():
            t = q.read_text()
            if '"xpu"' in t or '"cuda"' in t:
                q.write_text(t.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))
    p = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    p.diffusion.num_inference_steps = vis500.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return p, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200, help="rollouts par condition (200 suffit pour un diagnostic)")
    a = ap.parse_args()

    ckpt = f"results/runs/can/{a.run}/cooldown/checkpoints/005000/pretrained_model"
    if not Path(ckpt, "model.safetensors").exists():
        print(f"checkpoint absent : {ckpt}"); return 1
    keys = KEYS[a.run]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()[:a.n]
    policy, pre, post = load(ckpt, device)

    results = {}
    for mask in [None] + list(keys.values()):
        vis500.MASK_CAM = None if mask is None else mask.split(".")[-1]
        lab = "aucune (modèle complet)" if mask is None else f"sans {mask.split('.')[-1]}"
        print(f"\n##### masque : {lab} #####", flush=True)
        per, _ = vis500.rollout_can(policy, pre, post, states, device, keys)
        k = sum(e["success"] for e in per)
        results[lab] = {"n": len(per), "n_success": k, "rate": k / len(per)}
        print(f"=> {lab} : {k}/{len(per)} = {k/len(per):.1%}", flush=True)
    vis500.MASK_CAM = None

    base = results["aucune (modèle complet)"]["rate"]
    print(f"\n=== DIAGNOSTIC — {a.run} ===")
    for lab, r in results.items():
        if lab.startswith("aucune"):
            continue
        d = (r["rate"] - base) * 100
        verdict = ("caméra IGNORÉE par le modèle (effondrement)" if abs(d) < 5
                   else "caméra réellement utilisée")
        print(f"  {lab:28s} {r['rate']:6.1%}  ({d:+.1f} pts vs complet)  -> {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_r = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_r[a.run] = results
    OUT.write_text(json.dumps(all_r, indent=2))
    print(f"\nJSON : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
