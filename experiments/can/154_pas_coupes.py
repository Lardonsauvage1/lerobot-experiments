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
    C = {"paire": None, "tenu": 0, "pick": 0, "neuf": True, "n_inf": 0,
         "passes": 0, "evitees": 0, "choix": [0, 0], "age": []}
    rng = np.random.default_rng(0)

    def prep(batch):
        S = batch["observation.state"].shape[1]
        import einops
        imgs = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        st = batch["observation.state"]
        # ⚠️ La policy attend DEUX pas d'observation CONSÉCUTIFS par caméra. Une caméra
        # rafraîchie doit donc fournir une PAIRE fraîche (ses 2 images du pas courant), et
        # une caméra écartée une PAIRE ANCIENNE mais COHÉRENTE — surtout pas deux fois la
        # même image, ce qui lui présenterait une vue figée qu'elle n'a jamais vue à
        # l'entraînement. (Première version : prev=cur pour la vue écartée -> 18,5 % au lieu
        # de 82 %, artefact de structure temporelle et non coût du cache.)
        C["n_inf"] += 1

        def encode(i):
            f = dm.rgb_encoder[i](imgs[i])            # les 2 pas d'un coup
            return einops.rearrange(f, "(b s) d -> b s d", b=1, s=S)
        if C["neuf"]:
            C["paire"] = [encode(i) for i in range(2)]
            C["passes"] += 2 * S
            C["neuf"] = False
            C["age"] = [0, 0]
        else:
            if C["tenu"] <= 0 and a.tenir > 0:
                if a.regle == "hasard":
                    C["pick"] = int(rng.integers(0, 2))
                else:
                    cur = [p[:, -1] for p in C["paire"]]
                    g = torch.softmax(mlp(torch.cat(cur + [st[:, 0]], dim=-1)), dim=-1)
                    C["pick"] = int(g[0].argmax())
                C["tenu"] = a.tenir
                C["choix"][C["pick"]] += 1
            if a.tenir == 0:                          # témoin : tout rafraîchi
                C["paire"] = [encode(i) for i in range(2)]
                C["passes"] += 2 * S
                C["age"] = [0, 0]
            else:
                C["tenu"] -= 1
                i = C["pick"]
                C["paire"] = [encode(i) if j == i else C["paire"][j] for j in range(2)]
                C["passes"] += S                      # SEULE cette caméra est encodée
                C["evitees"] += S
                C["age"][i] = 0
                C["age"][1 - i] += 1
        return torch.cat([st] + C["paire"], dim=-1).flatten(start_dim=1)
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
    n_inf = C["n_inf"]                            # compté directement
    naif = 4 * n_inf
    print(f"\n  === k={a.tenir} · règle {a.regle} ===")
    print(f"  réussite : {k_}/{n_} = {k_/n_:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  passes d'encodeur : {C['passes']} pour {n_inf:.0f} inférences "
          f"({C['passes']/max(n_inf,1):.1f} par inférence, contre 4 en naïf) "
          f"→ {(1-C['passes']/max(naif,1))*100:.0f} % du calcul d'encodage supprimé")
    # 1 inférence = n_action_steps = 8 pas d'env ; contrôle Robomimic à 20 Hz -> 50 ms/pas.
    # (mesuré : 7,80 pas d'env par inférence). L'ancienne conversion 66,7 ms/inférence
    # supposait 1 inférence = 1 pas à 15 Hz : elle sous-estimait les durées d'un facteur 6.
    MS_PAR_INF = 8 * 50
    print(f"  âge médian de la vue la plus périmée : {np.median(ages):.1f} inférences "
          f"({np.median(ages)*MS_PAR_INF/1000:.1f} s)  ·  max {max(ages)} "
          f"({max(ages)*MS_PAR_INF/1000:.1f} s)")
    c = C["choix"]; s = max(sum(c), 1)
    print(f"  décisions : vue globale {c[0]/s*100:.0f} % · poignet {c[1]/s*100:.0f} %")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[tag] = {"n": n_, "n_success": k_, "success_rate": k_/n_, "ci95": [lo, hi],
                "tenir": a.tenir, "regle": a.regle, "passes": C["passes"], "evitees": C["evitees"],
                "age_median": float(np.median(ages)), "age_max": int(max(ages)), "choix": c,
                "passes_par_inference": C["passes"] / max(n_inf, 1),
                "economie_vs_naif": 1 - C["passes"] / max(naif, 1),
                "per_episode": [{"ep": i, "success": bool(x)} for i, x in enumerate(per_ep)]}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
