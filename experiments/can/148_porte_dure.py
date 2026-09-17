#!/usr/bin/env python
"""PORTE DURE : n'encoder qu'UNE caméra par pas, décidée depuis l'image PRÉCÉDENTE.

Le principe (celui de HCMS, arXiv 2104.09760) : la décision se prend sur des features
DÉJÀ calculées — celles du pas t-1 — donc sans rien encoder d'avance. La caméra non
retenue au pas t n'est pas encodée du tout : on RÉUTILISE ses features de t-1.

Ce que ça mesure :
  - le coût en réussite du durcissement (porte douce -> dure)
  - l'économie réelle : fraction des passes d'encodeur évitées

Aucun réentraînement : on durcit un modèle qui a déjà une porte (cam2_K_frozen_router).

Usage : venv312/bin/python experiments/can/148_porte_dure.py --n 500
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

CK = "results/runs/can/cam2_K_frozen_router/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/porte_dure.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mode", choices=["dure", "douce", "entrainee"], default="dure")
    ap.add_argument("--ckpt", default=CK)
    ap.add_argument("--tag", default=None, help="clé de sortie dans le JSON")
    a = ap.parse_args()
    CKPT = a.ckpt
    TAG = a.tag or a.mode

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sd = torch.load(Path(CKPT) / "camrouter.pt", map_location=device)
    hid, in_dim = sd["mlp.0.weight"].shape
    mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
                              torch.nn.Linear(hid, 2)).to(device)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    gamma = sd["gamma"].to(device)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(CKPT) / f; t = q.read_text()
        if '"xpu"' in t or '"cuda"' in t:
            q.write_text(t.replace('"xpu"', f'"{device.type}"').replace('"cuda"', f'"{device.type}"'))
    pol = DiffusionPolicy.from_pretrained(CKPT).to(device).eval()
    pol.diffusion.num_inference_steps = V.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    import einops
    dm = pol.diffusion
    stats = {"passes": 0, "evitees": 0, "choix": [0, 0]}
    HARD = a.mode in ("dure", "entrainee")
    TRAINED = a.mode == "entrainee"

    def prep(batch):
        B, S = batch["observation.state"].shape[:2]
        imgs = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        st = batch["observation.state"]
        # t-1 : on encode les 2 caméras (features déjà nécessaires, et c'est ce qui décide)
        prev = [dm.rgb_encoder[i](imgs[i][0::S]) for i in range(2)]      # (B, D)
        stats["passes"] += 2
        g_prev = torch.softmax(mlp(torch.cat(prev + [st[:, 0]], dim=-1)), dim=-1)
        if HARD:
            pick = int(torch.argmax(g_prev[0]))
            stats["choix"][pick] += 1
            cur = []
            for i in range(2):
                if i == pick:
                    cur.append(dm.rgb_encoder[i](imgs[i][1::S])); stats["passes"] += 1
                else:
                    cur.append(prev[i]); stats["evitees"] += 1        # RÉUTILISATION
        else:
            cur = [dm.rgb_encoder[i](imgs[i][1::S]) for i in range(2)]
            stats["passes"] += 2
        per = [torch.stack([prev[i], cur[i]], dim=1) for i in range(2)]  # (B, S, D)
        if TRAINED:
            # porte apprise DURE : la sélection est déjà appliquée (cur = prev pour la vue
            # écartée). Rien à pondérer — on reproduit `_prep_hard` de l'entraînement.
            return torch.cat([st] + per, dim=-1).flatten(start_dim=1)
        g = torch.softmax(mlp(torch.cat(per + [st], dim=-1)), dim=-1)
        per = [f * (1.0 + gamma * (2 * g[..., i:i+1] - 1.0)) for i, f in enumerate(per)]
        return torch.cat([st] + per, dim=-1).flatten(start_dim=1)
    dm._prepare_global_conditioning = prep

    states = V.make_eval_states()[:a.n]
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    per_ep, t0 = [], __import__("time").time()
    for i0 in range(0, len(states), 50):
        env = can_eval.make_env()
        for ep in range(i0, min(i0 + 50, len(states))):
            obs = env.reset_to(dict(states=states[ep])); pol.reset()
            ok = False
            for _ in range(can_eval.MAX_STEPS):
                imgs = {}
                for cam, k in keys.items():
                    im = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                    imgs[k] = torch.from_numpy(im.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
                s = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
                with torch.no_grad():
                    act = pol.select_action(pre({**imgs, "observation.state": s}))
                act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
                obs = env.step(act)[0]
                if env.is_success()["task"]: ok = True; break
            per_ep.append(ok)
        try: env.env.close()
        except Exception: pass
        del env
        print(f"  [chunk {i0}-{i0+50}] cumul {sum(per_ep)}/{len(per_ep)} = {sum(per_ep)/len(per_ep):.1%}", flush=True)

    k = sum(per_ep); n = len(per_ep)
    lo, hi = V.wilson_ci(k, n)
    tot = stats["passes"] + stats["evitees"]
    print(f"\n  === PORTE {a.mode.upper()} ===")
    print(f"  réussite : {k}/{n} = {k/n:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  passes d'encodeur : {stats['passes']} exécutées, {stats['evitees']} ÉVITÉES "
          f"({stats['evitees']/max(tot,1)*100:.0f} % du total)")
    if HARD:
        c = stats["choix"]; s = max(sum(c), 1)
        print(f"  caméra choisie : vue globale {c[0]/s*100:.0f} %  ·  poignet {c[1]/s*100:.0f} %")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[TAG] = {"n": n, "n_success": k, "success_rate": k/n, "ci95": [lo, hi],
                   "passes": stats["passes"], "evitees": stats["evitees"], "choix": stats["choix"],
                   "per_episode": [{"ep": i, "success": bool(x)} for i, x in enumerate(per_ep)]}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
