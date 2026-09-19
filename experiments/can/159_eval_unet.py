#!/usr/bin/env python
"""Éval 500 rollouts d'un modèle bi-caméra (vue extérieure + poignet), par checkpoint.

Sert au balayage de taille de U-Net : les modèles n'ont pas de routeur, donc on passe
par le moteur standard de 10_vision_500_rollouts.py, mêmes 500 états figés que toute
la campagne — les chiffres sont donc directement comparables aux 86,2 % de [64,128,256].
"""
import argparse, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np, torch

spec = importlib.util.spec_from_file_location("v", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
OUT = Path("results/runs/can/balayage_unet.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(a.ckpt) / f; t = q.read_text()
        if '"xpu"' in t or '"cuda"' in t:
            q.write_text(t.replace('"xpu"', f'"{dev.type}"').replace('"cuda"', f'"{dev.type}"'))
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    pol = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval()
    pol.diffusion.num_inference_steps = V.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    keys = {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}
    states = V.make_eval_states()[:a.n]
    per, elapsed = V.rollout_can(pol, pre, post, states, dev, keys)
    k = sum(e["success"] for e in per); n = len(per)
    lo, hi = V.wilson_ci(k, n)
    cfg = json.loads((Path(a.ckpt) / "config.json").read_text())
    print(f"\n  === {a.tag} — U-Net {cfg.get('down_dims')} ===")
    print(f"  réussite : {k}/{n} = {k/n:.1%} [{lo:.1%}, {hi:.1%}]   ({elapsed/60:.0f} min)")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[a.tag] = {"n": n, "n_success": k, "success_rate": k/n, "ci95": [lo, hi],
                  "down_dims": cfg.get("down_dims"), "ckpt": a.ckpt,
                  "per_episode": [{"ep": i, "success": bool(e["success"])} for i, e in enumerate(per)]}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
