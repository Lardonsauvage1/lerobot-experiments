#!/usr/bin/env python
"""ÉTAPE 0 du chantier mémoire — courbe de robustesse à l'occlusion de la cible.

Question : de combien l'occlusion de la canette fait-elle chuter une policy MARKOVIENNE,
et à partir de quel seuil ? C'est la mesure de référence contre laquelle tout mécanisme
de mémoire sera jugé (cf. docs/recherche/MEMOIRE.md).

Modèle témoin : `wristcap_A_agentview` (cooldown 5k) — mono-caméra agentview, ResNet34 +
U-Net[128,256,512], n_obs_steps=2, horizon 16 / 8 exécutées. C'est la MÊME structure que le
modèle déployé sur le vrai bras, donc ce qu'on mesure ici se transpose.
Référence connue sans occlusion : 337/500 = 67,4 % (results/runs/can/wristcap_eval500.json).

⚠️ Le point `radius = 0` est un TEST DE NON-RÉGRESSION : mêmes 500 états figés, même
modèle, rendu inchangé -> il doit retomber sur ~67 %. S'il s'en écarte, c'est le harnais
qui a bougé, pas l'occlusion.

Occlusion : la canette est retirée de la scène et la scène est re-rendue (aucun artefact
hors-distribution). Déclenchement quand le préhenseur est au-dessus d'elle à moins de
`radius` m en XY. Voir src/can_occlusion.py.

Sortie : results/runs/can/occluded/occlusion_curve.json (incrémental, reprise possible)
Lancer : nohup venv312/bin/python -u experiments/can/111_occlusion_curve.py \
           > results/logs/can/111_occlusion_curve.log 2>&1 &
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import numpy as np
import torch

from src import can_eval, can_occlusion

# réutilise les 500 états figés + le Wilson du harnais existant
spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis500)

CKPT = "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/occluded/occlusion_curve.json")
N_EVAL = 250          # IC95 ~±6 pts — assez pour une courbe ; on resserrera à 500 si besoin
STEPS = 10            # pas de diffusion à l'inférence (protocole habituel)
CHUNK = 50            # env recréé toutes les 50 ép. (anti-dégradation renderer)
RENDER_SIZE = 96

# rayon d'occlusion en mètres. 0 = témoin (jamais) ; 1e9 = canette jamais visible (borne basse)
# Grille RÉVISÉE le 2026-09-09 après le résultat à 4 cm (68,8 % -> ~24 %) : le plus PETIT
# rayon de la grille initiale consommait déjà les deux tiers du succès, donc 7/10/15 cm
# n'auraient lu qu'un plancher, trois fois. On les remplace par 1, 2, 3 cm pour capturer la
# PENTE — le régime où l'occlusion est brève et tardive, c'est-à-dire celui du robot réel.
# ∞ est conservé : borne basse absolue, indispensable comme référence.
RADII = [0.0, 0.01, 0.02, 0.03, 0.04, 1e9]

# surcharges pour le smoke test : OCC_N=4 OCC_RADII=0,1e9 OCC_OUT=/tmp/smoke.json
import os
N_EVAL = int(os.environ.get("OCC_N", N_EVAL))
if os.environ.get("OCC_RADII"):
    RADII = [float(x) for x in os.environ["OCC_RADII"].split(",")]
if os.environ.get("OCC_OUT"):
    OUT = Path(os.environ["OCC_OUT"])


def rollout_occluded(policy, pre, post, states, device, radius):
    """Moteur de rollout Can + occlusion synthétique + sonde. Calqué sur 10_vision_500."""
    render_fn = can_occlusion.make_render_fn(radius)
    probe = can_occlusion.OcclusionProbe(radius)
    rec_traj = can_occlusion.TrajectoryRecorder()
    per_all = []
    t0 = time.time()
    for i in range(0, len(states), CHUNK):
        env = can_eval.make_env()
        for ep in range(i, min(i + CHUNK, len(states))):
            obs = env.reset_to(dict(states=states[ep]))
            policy.reset()
            probe.start_episode()
            rec_traj.start_episode()
            t_success = None
            for step_i in range(can_eval.MAX_STEPS):
                probe.observe(env, obs)
                img = render_fn(env, obs, RENDER_SIZE)
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                state_t = torch.from_numpy(vis500.state_proprio(obs))
                obs_dict = pre({"observation.image": img_t.unsqueeze(0).to(device),
                                "observation.state": state_t.unsqueeze(0).to(device)})
                with torch.no_grad():
                    a = policy.select_action(obs_dict)
                a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
                # enregistré AVANT le step : proprio et can_pos de l'instant où l'action est décidée
                rec_traj.step(vis500.state_proprio(obs), a,
                              np.asarray(obs["object"]).flatten()[7:10])
                obs = env.step(a)[0]
                if env.is_success()["task"]:
                    t_success = step_i
                    break
            rec = {"success": t_success is not None, "t_success": t_success}
            rec.update(probe.summary(rec["success"]))
            rec_traj.end_episode(probe, {"ep": ep, "success": rec["success"],
                                         "t_success": t_success,
                                         "n_approaches": rec["n_approaches"],
                                         "n_occl_events": rec["n_occl_events"]})
            per_all.append(rec)
        try:
            env.env.close()
        except Exception:
            pass
        del env
        k = sum(e["success"] for e in per_all)
        print(f"    [{i}-{i + CHUNK}] cumul {k}/{len(per_all)} = {k / len(per_all):.1%}", flush=True)
    tag = "inf" if radius >= 1e3 else f"{int(round(radius * 100)):02d}"
    path = rec_traj.save(OUT.parent / f"traj_r{tag}.npz", radius=radius, ckpt=CKPT)
    print(f"    trajectoires -> {path} ({rec_traj._n} pas)", flush=True)
    return per_all, time.time() - t0


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()[:N_EVAL]
    print(f"États : {states.shape} | device {device}", flush=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    policy = DiffusionPolicy.from_pretrained(CKPT).to(device).eval()
    policy.diffusion.num_inference_steps = STEPS
    pre = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text())["results"] if OUT.exists() else []
    done = {r["radius"] for r in results}

    for radius in RADII:
        if radius in done:
            print(f"\nradius={radius} déjà fait, skip", flush=True)
            continue
        label = "témoin (jamais occlus)" if radius == 0 else (
            "jamais visible" if radius >= 1e3 else f"{radius * 100:.0f} cm")
        print(f"\n##### radius = {radius}  [{label}] #####", flush=True)
        per, elapsed = rollout_occluded(policy, pre, post, states, device, radius)
        k = sum(e["success"] for e in per); n = len(per)
        lo, hi = vis500.wilson_ci(k, n)
        t_succs = [e["t_success"] for e in per if e["success"]]
        rec = {"radius": radius, "label": label, "n": n, "n_success": k,
               "success_rate": k / n, "ci95": [lo, hi],
               "t_success_median": float(np.median(t_succs)) if t_succs else None,
               "elapsed_min": elapsed / 60}
        rec.update(can_occlusion.aggregate_occlusion(per))
        results.append(rec)
        results.sort(key=lambda r: r["radius"])
        OUT.write_text(json.dumps({"ckpt": CKPT, "n_eval": N_EVAL, "steps": STEPS,
                                   "results": results}, indent=2))
        rr = rec["recovery_rate"]
        rr_txt = f"{rr:.1%}" if rr is not None else "-"
        print(f"\n=> radius={radius} [{label}] : {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"
              f" | occlus {rec['n_episodes_occluded']} ép. | récup {rr_txt}", flush=True)
        print(f"   temps occlus {rec['occl_fraction_mean']:.1%} | approches/ép "
              f"{rec['n_approaches_mean']:.2f} (échecs : {rec['n_approaches_mean_failed']})"
              f" | {elapsed / 60:.1f} min", flush=True)

    print("\n=== COURBE D'OCCLUSION ===", flush=True)
    print(f"{'rayon':>10} {'label':>22} {'succès':>18} {'récup':>8} {'appr/ép':>9}")
    for r in results:
        rr = f"{r['recovery_rate']:.1%}" if r["recovery_rate"] is not None else "  -"
        print(f"{r['radius']:>10} {r['label']:>22} {r['success_rate']:>7.1%} "
              f"[{r['ci95'][0]*100:4.1f}-{r['ci95'][1]*100:4.1f}] {rr:>8} "
              f"{r['n_approaches_mean']:>9.2f}")
    print(f"\nJSON : {OUT}")


if __name__ == "__main__":
    main()
