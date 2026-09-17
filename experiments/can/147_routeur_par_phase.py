#!/usr/bin/env python
"""Comment la porte du routeur se comporte-t-elle AU FIL de la trajectoire ?

Trois questions :
  1. bascule-t-elle souvent, ou choisit-elle une vue et s'y tient ?
  2. une vue est-elle privilégiée selon la PHASE — approche, saisie, transport ?
  3. le profil temporel a-t-il une forme, ou est-ce du bruit ?

Découpage des phases, à partir de la vérité simulation :
  APPROCHE  : canette au repos, pince encore ouverte
  SAISIE    : commande de fermeture envoyée, canette pas encore soulevée
  TRANSPORT : canette soulevée (z > 0,90)

Usage : venv312/bin/python experiments/can/147_routeur_par_phase.py --run cam2_H_router --n 25
"""
import argparse
import importlib.util
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

CAN_LIFTED = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cam2_H_router")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    CK = f"results/runs/can/{a.run}/cooldown/checkpoints/005000/pretrained_model"

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sd = torch.load(Path(CK) / "camrouter.pt", map_location=device)
    hid, in_dim = sd["mlp.0.weight"].shape
    mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
                              torch.nn.Linear(hid, 2)).to(device)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    gamma = float(sd["gamma"])
    print(f"  {a.run} — gamma={gamma:.4f}\n", flush=True)

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
    pol.diffusion.num_inference_steps = 10
    pre = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    import einops
    dm = pol.diffusion
    cur = {}

    def prep(batch):
        B, S = batch["observation.state"].shape[:2]
        imgs = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        per = [einops.rearrange(e(i), "(b s) d -> b s d", b=B, s=S)
               for e, i in zip(dm.rgb_encoder, imgs, strict=True)]
        st = batch["observation.state"]
        g = torch.softmax(mlp(torch.cat(per + [st], dim=-1)), dim=-1)
        cur["g"] = float(g[0, -1, 0])          # poids de l'AGENTVIEW au pas courant
        per = [f * (1.0 + gamma * (2 * g[..., i:i+1] - 1.0)) for i, f in enumerate(per)]
        return torch.cat([st] + per, dim=-1).flatten(start_dim=1)
    dm._prepare_global_conditioning = prep

    states = V.make_eval_states()[:a.n]
    env = can_eval.make_env()
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    epis = []
    for ep in range(len(states)):
        obs = env.reset_to(dict(states=states[ep])); pol.reset()
        G, PH = [], []
        closed = False
        for _ in range(can_eval.MAX_STEPS):
            can_z = float(np.asarray(obs["object"]).flatten()[9])
            imgs = {}
            for cam, k in keys.items():
                im = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                imgs[k] = torch.from_numpy(im.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
            st = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = pol.select_action(pre({**imgs, "observation.state": st}))
            act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
            if act[6] > 0: closed = True
            PH.append("TRANSPORT" if can_z > CAN_LIFTED else ("SAISIE" if closed else "APPROCHE"))
            G.append(cur["g"])
            obs = env.step(act)[0]
            if env.is_success()["task"]: break
        epis.append((np.array(G), np.array(PH)))
        if (ep+1) % 5 == 0: print(f"    {ep+1}/{len(states)}", flush=True)

    allG = np.concatenate([g for g, _ in epis])
    allP = np.concatenate([p for _, p in epis])
    print(f"\n  === PAR PHASE (poids donné à la VUE GLOBALE ; 1-x = poignet) ===")
    for ph in ("APPROCHE", "SAISIE", "TRANSPORT"):
        m = allP == ph
        if m.sum():
            print(f"    {ph:10s} {m.sum():5d} pas ({m.mean()*100:4.1f} %)   "
                  f"vue globale {allG[m].mean():.3f} ± {allG[m].std():.3f}")
    print(f"\n  === BASCULE-T-ELLE ? (vue dominante = celle de poids > 0,5) ===")
    flips, steps = 0, 0
    for g, _ in epis:
        d = (g > 0.5).astype(int)
        flips += int(np.sum(d[1:] != d[:-1])); steps += len(d) - 1
    print(f"    {flips} changements de vue dominante sur {steps} pas  -> une bascule tous les {steps/max(flips,1):.0f} pas")
    print(f"    vue globale dominante sur {(allG>0.5).mean()*100:.0f} % des pas")
    print(f"\n  === PROFIL TEMPOREL (épisode normalisé en 10 tranches) ===")
    prof = np.zeros(10); cnt = np.zeros(10)
    for g, _ in epis:
        idx = (np.linspace(0, 0.999, len(g)) * 10).astype(int)
        for i, v in zip(idx, g): prof[i] += v; cnt[i] += 1
    prof /= np.maximum(cnt, 1)
    for i, v in enumerate(prof):
        bar = "█" * int(round(v * 40))
        print(f"    {i*10:3d}-{(i+1)*10:3d} %  {v:.3f}  {bar}")


if __name__ == "__main__":
    main()
