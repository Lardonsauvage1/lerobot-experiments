#!/usr/bin/env python
"""Jusqu'où peut-on COUPER une caméra avant que ça coûte — et le choix se met-il à compter ?

Le constat qui motive ce test : à UN pas de retard, le hasard égale la porte apprise
(411/500 des deux côtés, p=1,000), parce que la porte ne PRIVE jamais le modèle d'une vue,
elle en RETARDE une de 67 ms. Ici on laisse le retard s'accumuler.

Ce qui change par rapport à 148 : un VRAI CACHE entre pas de contrôle. Les features de
chaque caméra sont mémorisées ; à chaque pas on n'en RAFRAÎCHIT qu'une, l'autre garde les
siennes aussi longtemps qu'elle n'est pas rechoisie. `--tenir k` maintient la décision k pas
d'affilée, donc la vue écartée vieillit jusqu'à k pas (67 ms × k).

Le cache fait aussi tomber le coût réel : 1 passe d'encodeur par pas au lieu de 4
(2 caméras × 2 pas d'historique, tout recalculé à chaque fois dans l'implémentation naïve).

Deux règles comparées à chaque k :
  apprise — l'argmax de la porte entraînée
  hasard  — tirage au sort
Si l'écart apprise-hasard grandit avec k, la politique par phase a enfin un levier.

Usage : venv312/bin/python experiments/can/154_pas_coupes.py --tenir 6 --regle apprise --n 200
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

CK = "results/runs/can/cam2_L_hardgate_s42/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/pas_coupes.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tenir", type=int, default=1, help="pas pendant lesquels la décision tient")
    ap.add_argument("--regle", choices=["apprise", "hasard"], default="apprise")
    a = ap.parse_args()
    tag = f"k{a.tenir}_{a.regle}"

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sd = torch.load(Path(CK) / "camrouter.pt", map_location=device)
    hid, in_dim = sd["mlp.0.weight"].shape
    mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
                              torch.nn.Linear(hid, 2)).to(device)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()

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

    dm = pol.diffusion
    C = {"prev": None, "cur": None, "tenu": 0, "pick": 0, "neuf": True,
         "passes": 0, "evitees": 0, "choix": [0, 0], "age": []}
    rng = np.random.default_rng(0)

    def prep(batch):
        S = batch["observation.state"].shape[1]
        import einops
        imgs = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        st = batch["observation.state"]
        if C["neuf"]:                       # 1er pas de l'épisode : rien en cache
            f = [dm.rgb_encoder[i](imgs[i][1::S]) for i in range(2)]
            C["passes"] += 2
            C["prev"], C["cur"] = [x.clone() for x in f], f
            C["neuf"] = False
            C["age"] = [0, 0]
        else:
            if C["tenu"] <= 0:              # on redécide
                if a.regle == "hasard":
                    C["pick"] = int(rng.integers(0, 2))
                else:
                    g = torch.softmax(mlp(torch.cat(C["cur"] + [st[:, 0]], dim=-1)), dim=-1)
                    C["pick"] = int(g[0].argmax())
                C["tenu"] = a.tenir
                C["choix"][C["pick"]] += 1
            C["tenu"] -= 1
            i = C["pick"]
            C["prev"] = list(C["cur"])
            new = list(C["cur"])
            new[i] = dm.rgb_encoder[i](imgs[i][1::S])      # SEULE caméra encodée
            C["cur"] = new
            C["passes"] += 1; C["evitees"] += 1
            C["age"][i] = 0
            C["age"][1 - i] += 1
        per = [torch.stack([C["prev"][i], C["cur"][i]], dim=1) for i in range(2)]
        return torch.cat([st] + per, dim=-1).flatten(start_dim=1)
    dm._prepare_global_conditioning = prep

    states = V.make_eval_states()[:a.n]
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    per_ep, ages = [], []
    for i0 in range(0, len(states), 50):
        env = can_eval.make_env()
        for ep in range(i0, min(i0 + 50, len(states))):
            obs = env.reset_to(dict(states=states[ep])); pol.reset()
            C["neuf"] = True; C["tenu"] = 0        # cache vidé à chaque épisode
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
                ages.append(max(C["age"]))
                obs = env.step(act)[0]
                if env.is_success()["task"]: ok = True; break
            per_ep.append(ok)
        try: env.env.close()
        except Exception: pass
        del env
        print(f"  [chunk {i0}] cumul {sum(per_ep)}/{len(per_ep)} = {sum(per_ep)/len(per_ep):.1%}", flush=True)

    k_, n_ = sum(per_ep), len(per_ep)
    lo, hi = V.wilson_ci(k_, n_)
    # référence honnête = l'implémentation NAÏVE : 2 caméras × 2 pas d'historique réencodés
    # à chaque pas de contrôle, soit 4 passes. Le cache seul en supprime déjà 2.
    n_pas = C["passes"] + C["evitees"]
    naif = 4 * n_pas
    print(f"\n  === k={a.tenir} · règle {a.regle} ===")
    print(f"  réussite : {k_}/{n_} = {k_/n_:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  passes d'encodeur : {C['passes']} pour {n_pas} pas "
          f"({C['passes']/max(n_pas,1):.2f} par pas, contre 4 en naïf) "
          f"→ {(1-C['passes']/max(naif,1))*100:.0f} % du calcul d'encodage supprimé")
    print(f"  âge médian de la vue la plus périmée : {np.median(ages):.1f} pas "
          f"({np.median(ages)*66.7:.0f} ms)  ·  max {max(ages)}")
    c = C["choix"]; s = max(sum(c), 1)
    print(f"  décisions : vue globale {c[0]/s*100:.0f} % · poignet {c[1]/s*100:.0f} %")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[tag] = {"n": n_, "n_success": k_, "success_rate": k_/n_, "ci95": [lo, hi],
                "tenir": a.tenir, "regle": a.regle, "passes": C["passes"], "evitees": C["evitees"],
                "age_median": float(np.median(ages)), "age_max": int(max(ages)), "choix": c,
                "passes_par_pas": C["passes"] / max(n_pas, 1),
                "economie_vs_naif": 1 - C["passes"] / max(naif, 1),
                "per_episode": [{"ep": i, "success": bool(x)} for i, x in enumerate(per_ep)]}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
