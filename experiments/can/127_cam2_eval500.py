#!/usr/bin/env python
"""CAM2 — éval 500 rollouts des deux bras, AVEC le détail par épisode.

Pourquoi ce script et pas `10_vision_500_rollouts.py` : celui-ci ne sauvegardait que les
agrégats. Résultat, la comparaison historique « côté seule 70,4 % vs côté+dessus 74,8 % »
est **non appariée** — ses IC95 se chevauchent et elle ne tranche rien, alors que les deux
évals partageaient pourtant les MÊMES 500 états initiaux figés.

Ici on écrit `per_episode` (succès + t_success par épisode), ce qui autorise un McNemar
apparié — le seul test valide quand les deux bras voient les mêmes états.

Sortie : results/runs/can/cam2_eval500.json  (+ résumé apparié en fin de run)
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis500)   # charge robosuite + le moteur de rollout

import math
from math import comb

import numpy as np
import torch

OUT = Path("results/runs/can/cam2_eval500.json")

MODELS = [
    {"name": "A_side", "label": "côté seule (agentview)",
     "ckpt": "results/runs/can/cam2_A_side/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.image"}},
    {"name": "B_side_top", "label": "côté + dessus (agentview + birdview)",
     "ckpt": "results/runs/can/cam2_B_side_top/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "birdview": "observation.images.birdview"}},
]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_report(per_a, per_b):
    a = np.array([e["success"] for e in per_a], dtype=bool)
    b = np.array([e["success"] for e in per_b], dtype=bool)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    only_b = int((b & ~a).sum())     # le 2e capteur gagne
    only_a = int((a & ~b).sum())     # le 2e capteur perd
    d = b.mean() - a.mean()
    var = (only_a + only_b - (only_a - only_b) ** 2 / n) / n ** 2
    h = 1.96 * math.sqrt(var)
    p = mcnemar_exact(only_b, only_a)
    return {"n": n, "k_a": int(a.sum()), "k_b": int(b.sum()),
            "delta": d, "lo": d - h, "hi": d + h, "p": p,
            "gagnes_par_2e_cam": only_b, "perdus_par_2e_cam": only_a}


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()

    results = json.loads(OUT.read_text())["results"] if OUT.exists() else []
    done = {r["name"] for r in results}

    for m in MODELS:
        if m["name"] in done:
            print(f"\n{m['name']} déjà fait, skip", flush=True)
            continue
        if not Path(m["ckpt"], "model.safetensors").exists():
            print(f"\n⚠️  {m['name']} : checkpoint absent ({m['ckpt']})", flush=True)
            continue
        print(f"\n##### {m['label']} #####", flush=True)
        policy, pre, post = _load(m["ckpt"], device)
        per, elapsed = vis500.rollout_can(policy, pre, post, states, device, m["image_keys"])
        k = sum(e["success"] for e in per)
        lo, hi = vis500.wilson_ci(k, len(per))
        results.append({"name": m["name"], "label": m["label"], "ckpt": m["ckpt"],
                        "n": len(per), "n_success": k, "success_rate": k / len(per),
                        "ci95": [lo, hi], "elapsed_min": elapsed / 60,
                        "per_episode": [{"ep": i, "success": bool(e["success"]),
                                         "t_success": e["t_success"]} for i, e in enumerate(per)]})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"n_eval": len(states), "results": results}, indent=2))
        print(f"\n=> {m['label']} : {k}/{len(per)} = {k/len(per):.1%} "
              f"[{lo:.1%}, {hi:.1%}]  ({elapsed/60:.1f} min)", flush=True)
        del policy

    by = {r["name"]: r for r in results}
    if "A_side" in by and "B_side_top" in by:
        r = paired_report(by["A_side"]["per_episode"], by["B_side_top"]["per_episode"])
        print("\n=== COMPARAISON APPARIÉE (McNemar exact) ===")
        print(f"  côté seule    : {r['k_a']}/{r['n']} = {r['k_a']/r['n']:.1%}")
        print(f"  côté + dessus : {r['k_b']}/{r['n']} = {r['k_b']/r['n']:.1%}")
        print(f"  apport de la 2e caméra : {r['delta']*100:+.1f} pts "
              f"IC95 [{r['lo']*100:+.1f} ; {r['hi']*100:+.1f}]   p = {r['p']:.4g}"
              f"{'  ★ significatif' if r['p'] < 0.05 else '  (non significatif)'}")
        print(f"  épisodes gagnés par la 2e caméra : {r['gagnes_par_2e_cam']} · "
              f"perdus : {r['perdus_par_2e_cam']}")
        payload = json.loads(OUT.read_text())
        payload["paired"] = r
        OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON : {OUT}")


def _fix_device(ckpt, device):
    """Un checkpoint entraîné sur atomman embarque device='xpu' dans ses JSON de pipeline ;
    la même chose vaut pour 'cuda' depuis gb10. Réécriture idempotente avant chargement."""
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(ckpt) / f
        if not q.exists():
            continue
        txt = q.read_text()
        if '"xpu"' in txt or '"cuda"' in txt:
            q.write_text(txt.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))
            print(f"  device réécrit dans {f} -> {device.type}", flush=True)


def _load(ckpt, device):
    _fix_device(ckpt, device)
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    p = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    p.diffusion.num_inference_steps = vis500.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return p, pre, post


if __name__ == "__main__":
    main()
