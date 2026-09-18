#!/usr/bin/env python
"""Le modèle CORRIGE-T-IL sa position, ou exécute-t-il une trajectoire apprise ?

Hypothèse testée (formulée en regardant les vidéos d'échec) : « le bras n'essaie pas de
rectifier sa position comme sa caméra embarquée pourrait le lui permettre ; il va à une
position et il attrape, qu'il y ait la canette ou pas. »

Protocole : on TÉLÉPORTE la canette de `--delta` m en cours d'approche, au moment où la
pince passe sous 15 cm d'elle. Puis on regarde OÙ la pince se ferme.
  - si elle se ferme sur la NOUVELLE position  -> boucle fermée, la vision sert jusqu'au bout
  - si elle se ferme sur l'ANCIENNE            -> boucle ouverte, la vision ne sert qu'au départ

Témoin : mêmes épisodes, delta = 0. Tout écart se lit contre ce témoin, pas dans l'absolu.

Usage : venv312/bin/python experiments/can/152_test_boucle_fermee.py --n 50 --delta 0.04
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import numpy as np
import torch

from src import can_eval

spec = importlib.util.spec_from_file_location("v", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

CK = "results/runs/can/cam2_I2_frozen_s43/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/boucle_fermee.json")
DECLENCHE = 0.15     # distance pince-canette (m) qui déclenche la téléportation
CAN_REPOS = 0.8600   # z de la canette au repos : au-dessus = elle a bougé


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--delta", type=float, default=0.04, help="déplacement horizontal (m)")
    ap.add_argument("--masque", default=None, choices=["poignet", "exterieure"],
                    help="coupe une caméra (0 après normalisation = image moyenne) pour "
                         "savoir LAQUELLE porte la correction")
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(CK) / f; t = q.read_text()
        if '"xpu"' in t or '"cuda"' in t:
            q.write_text(t.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))
    pol = DiffusionPolicy.from_pretrained(CK).to(device).eval()
    pol.diffusion.num_inference_steps = V.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = V.make_eval_states()[:a.n]
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    suff = f"_sans_{a.masque}" if a.masque else ""
    boucle = (a.delta,) if a.masque else (0.0, a.delta)   # le témoin est déjà mesuré
    for delta in boucle:
        tag = ("temoin" if delta == 0 else f"perturbe_{int(delta*100)}cm") + suff
        print(f"\n  ##### {tag} (delta = {delta*100:.0f} cm) #####", flush=True)
        env = can_eval.make_env()
        sim = env.env.sim
        eps = []
        for ep in range(len(states)):
            obs = env.reset_to(dict(states=states[ep])); pol.reset()
            can0 = np.asarray(obs["object"]).flatten()[7:10].copy()
            can_now, bouge, closed = can0.copy(), False, False
            d_new = d_old = None
            ok = False
            for step in range(can_eval.MAX_STEPS):
                eef = np.asarray(obs["robot0_eef_pos"]).flatten()
                # téléportation UNE SEULE FOIS, quand la pince approche
                if (not bouge) and delta > 0 and np.linalg.norm(eef[:2] - can_now[:2]) < DECLENCHE:
                    q = sim.data.get_joint_qpos("Can_joint0").copy()
                    ang = np.random.default_rng(ep).uniform(0, 2 * np.pi)
                    q[0] += delta * np.cos(ang); q[1] += delta * np.sin(ang)
                    sim.data.set_joint_qpos("Can_joint0", q); sim.forward()
                    can_now = q[:3].copy(); bouge = True
                raw = {c: env.env.sim.render(height=96, width=96, camera_name=c)[::-1]
                       for c in keys}
                imgs = {k: torch.from_numpy(raw[c].copy()).permute(2, 0, 1).float()
                        .unsqueeze(0).to(device) / 255. for c, k in keys.items()}
                st = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
                batch = pre({**imgs, "observation.state": st})
                if a.masque:
                    # masquage APRÈS normalisation : 0 = image moyenne, signal « absent » neutre
                    mk = "wrist" if a.masque == "poignet" else "agentview"
                    for _k in list(batch):
                        if _k.startswith("observation.image") and mk in _k:
                            batch[_k] = torch.zeros_like(batch[_k])
                with torch.no_grad():
                    act = pol.select_action(batch)
                act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
                # OÙ la pince se ferme-t-elle, la 1re fois ?
                if act[6] > 0 and not closed:
                    closed = True
                    d_new = float(np.linalg.norm(eef[:2] - can_now[:2]))
                    d_old = float(np.linalg.norm(eef[:2] - can0[:2]))
                obs = env.step(act)[0]
                if env.is_success()["task"]: ok = True; break
            zmax = float(np.asarray(obs["object"]).flatten()[9])
            eps.append({"ep": ep, "success": ok, "teleporte": bool(bouge),
                        "d_nouvelle": d_new, "d_ancienne": d_old, "z_fin": zmax})
            if (ep + 1) % 10 == 0:
                s = sum(e["success"] for e in eps)
                print(f"    {ep+1}/{len(states)} — {s}/{len(eps)} = {s/len(eps):.0%}", flush=True)
        try: env.env.close()
        except Exception: pass

        k = sum(e["success"] for e in eps); n = len(eps)
        tp = [e for e in eps if e["teleporte"] and e["d_nouvelle"] is not None]
        print(f"\n    réussite : {k}/{n} = {k/n:.1%}", flush=True)
        if tp:
            dn = float(np.median([e["d_nouvelle"] for e in tp]))
            do = float(np.median([e["d_ancienne"] for e in tp]))
            suit = sum(1 for e in tp if e["d_nouvelle"] < e["d_ancienne"])
            print(f"    à la fermeture, distance médiane à la NOUVELLE position : {dn*100:.1f} cm")
            print(f"                    distance médiane à l'ANCIENNE position : {do*100:.1f} cm")
            print(f"    ferme plus près de la NOUVELLE : {suit}/{len(tp)} "
                  f"({suit/len(tp)*100:.0f} %)  -> {'CORRIGE' if suit/len(tp) > 0.6 else 'NE CORRIGE PAS'}")
        else:
            dd = [e["d_nouvelle"] for e in eps if e["d_nouvelle"] is not None]
            if dd:
                print(f"    précision de fermeture (témoin) : médiane {np.median(dd)*100:.1f} cm")
        res[tag] = {"n": n, "n_success": k, "success_rate": k / n, "delta": delta,
                    "per_episode": eps}
        OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
