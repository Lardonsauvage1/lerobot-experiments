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
    {"name": "C_side_wrist", "label": "côté + embarquée (agentview + poignet)",
     "ckpt": "results/runs/can/cam2_C_side_wrist/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "C2_wrist_fixstats", "label": "côté + embarquée, stats CORRIGÉES",
     "ckpt": "results/runs/can/cam2_C2_wrist_fixstats/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "C3_camdrop", "label": "côté + embarquée, stats corrigées + DROPOUT caméra",
     "ckpt": "results/runs/can/cam2_C3_camdrop/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "A2_fixstats", "label": "côté seule, stats CORRIGÉES",
     "ckpt": "results/runs/can/cam2_A2_fixstats/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.image"}},
    {"name": "B2_fixstats", "label": "côté + dessus, stats CORRIGÉES",
     "ckpt": "results/runs/can/cam2_B2_fixstats/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "birdview": "observation.images.birdview"}},
    {"name": "C4_auxhead", "label": "côté + embarquée, stats corrigées + TÊTE AUXILIAIRE",
     "ckpt": "results/runs/can/cam2_C4_auxhead/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "aux_s42", "label": "tête aux · graine 42",
     "ckpt": "results/runs/can/cam2_aux_s42/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "aux_s43", "label": "tête aux · graine 43",
     "ckpt": "results/runs/can/cam2_aux_s43/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "aux_s44", "label": "tête aux · graine 44",
     "ckpt": "results/runs/can/cam2_aux_s44/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "ref_s43", "label": "référence poignet · graine 43",
     "ckpt": "results/runs/can/cam2_ref_s43/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"agentview": "observation.images.agentview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
    {"name": "D_birdview_only", "label": "DESSUS seule (le bras la masque)",
     "ckpt": "results/runs/can/cam2_D_birdview_only/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"birdview": "observation.image"}},
    {"name": "E_birdview_wrist", "label": "DESSUS + embarquée (le cas réel)",
     "ckpt": "results/runs/can/cam2_E_birdview_wrist/cooldown/checkpoints/005000/pretrained_model",
     "image_keys": {"birdview": "observation.images.birdview",
                    "robot0_eye_in_hand": "observation.images.wrist"}},
]

# comparaisons à produire : (bras, référence). La référence est toujours « côté seule »,
# plus C vs B pour départager les deux façons d'ajouter un 2e capteur.
PAIRS = [("B_side_top", "A_side"), ("C_side_wrist", "A_side"), ("C_side_wrist", "B_side_top"),
         ("C2_wrist_fixstats", "C_side_wrist"),   # LE contrôle : la normalisation explique-t-elle la chute ?
         ("C2_wrist_fixstats", "A_side"),
         ("C3_camdrop", "C2_wrist_fixstats"),  # LE test : le dropout débloque-t-il le poignet ?
         ("C3_camdrop", "A_side"),
         ("A2_fixstats", "A_side"),          # les stats valent-elles aussi 7 pts en mono-caméra ?
         ("C2_wrist_fixstats", "A2_fixstats"),  # LA comparaison propre : poignet vs seule, tous deux corrigés
         ("C3_camdrop", "A2_fixstats"),
         ("B2_fixstats", "B_side_top"),     # la correction vaut-elle aussi pour le bi-caméra ?
         ("B2_fixstats", "A2_fixstats"),   # 2 caméras vs 1, tout corrigé
         ("C4_auxhead", "C2_wrist_fixstats"),  # LE test : la tête auxiliaire débloque-t-elle le poignet ?
         ("C4_auxhead", "C3_camdrop"),     # tête auxiliaire vs dropout, à armes égales
         ("C4_auxhead", "A2_fixstats"),   # reste-t-il un écart avec la caméra seule ?
         ("E_birdview_wrist", "D_birdview_only")]  # ⭐ LA question : le poignet aide-t-il
                                          # quand la caméra fixe EST masquée ? (cas réel)


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
    print("\n=== NIVEAUX (500 rollouts, états figés) ===")
    for m in MODELS:
        if m["name"] in by:
            r = by[m["name"]]
            print(f"  {r['label']:44s} {r['n_success']:3d}/{r['n']} = {r['success_rate']:6.1%} "
                  f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")

    pairs_out = {}
    header = False
    for tag, ref in PAIRS:
        if tag not in by or ref not in by:
            continue
        if not header:
            print("\n=== COMPARAISONS APPARIÉES (McNemar exact) ===")
            header = True
        r = paired_report(by[ref]["per_episode"], by[tag]["per_episode"])
        pairs_out[f"{tag}_vs_{ref}"] = r
        print(f"  {tag} vs {ref:12s} {r['delta']*100:+6.1f} pts  "
              f"IC95 [{r['lo']*100:+6.1f} ; {r['hi']*100:+6.1f}]  p = {r['p']:9.4g}"
              f"{'  ★' if r['p'] < 0.05 else ''}")
        print(f"      gagnés {r['gagnes_par_2e_cam']:3d} · perdus {r['perdus_par_2e_cam']:3d}"
              f"   (discordants = ce qui porte l'effet)")
    if pairs_out:
        payload = json.loads(OUT.read_text())
        payload["paired"] = pairs_out
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
