#!/usr/bin/env python
"""Que fait RÉELLEMENT la porte du routeur ? Diagnostic sur rollouts.

On rejoue des épisodes avec le modèle H (routeur) en enregistrant, à chaque pas :
  - les poids de la porte g = (g_agentview, g_poignet)
  - si la canette est géométriquement MASQUÉE pour l'agentview (vérité sim, jamais
    utilisée à l'entraînement — c'est un diagnostic, pas une béquille)

Trois questions :
  1. la porte varie-t-elle, ou est-elle figée ? (une porte constante = routage inutile)
  2. sur quelle caméra s'appuie-t-elle en moyenne ?
  3. se ferme-t-elle sur l'agentview quand celle-ci est effectivement masquée ?

Usage : venv312/bin/python experiments/can/140_diag_routeur.py --n 20
"""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from src import can_eval, can_occlusion

spec = importlib.util.spec_from_file_location("v", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

CKPT = "results/runs/can/cam2_H_router/cooldown/checkpoints/005000/pretrained_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--radius", type=float, default=0.03)
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sd = torch.load(Path(CKPT) / "camrouter.pt", map_location=device)
    hid, in_dim = sd["mlp.0.weight"].shape
    mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
                              torch.nn.Linear(hid, 2)).to(device)
    mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()
    gamma = float(sd["gamma"])
    print(f"  routeur : gamma={gamma:.4f}, MLP {in_dim}->{hid}->2\n", flush=True)

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
    pol.diffusion.num_inference_steps = 10
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    import einops
    dm = pol.diffusion
    gates, occl = [], []

    def prep(batch):
        B, S = batch["observation.state"].shape[:2]
        imgs = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        per = [einops.rearrange(e(i), "(b s) d -> b s d", b=B, s=S)
               for e, i in zip(dm.rgb_encoder, imgs, strict=True)]
        st = batch["observation.state"]
        g = torch.softmax(mlp(torch.cat(per + [st], dim=-1)), dim=-1)
        gates.append(g[:, -1].detach().cpu().numpy().ravel())
        per = [f * (1.0 + gamma * (2 * g[..., i:i+1] - 1.0)) for i, f in enumerate(per)]
        return torch.cat([st] + per, dim=-1).flatten(start_dim=1)
    dm._prepare_global_conditioning = prep

    states = V.make_eval_states()[:a.n]
    env = can_eval.make_env()
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    for ep in range(len(states)):
        obs = env.reset_to(dict(states=states[ep])); pol.reset()
        for _ in range(can_eval.MAX_STEPS):
            eef = np.asarray(obs["robot0_eef_pos"]).flatten()
            can = np.asarray(obs["object"]).flatten()[7:10]
            occl.append(can_occlusion.occluded_from_xyz(eef, can, a.radius))
            imgs = {}
            for cam, k in keys.items():
                im = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                imgs[k] = torch.from_numpy(im.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
            st = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = pol.select_action(pre({**imgs, "observation.state": st}))
            act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
            obs = env.step(act)[0]
            if env.is_success()["task"]: break
        if (ep+1) % 5 == 0: print(f"    {ep+1}/{len(states)} épisodes", flush=True)

    G = np.stack(gates)[:len(occl)]          # (T, 2) : [agentview, poignet]
    O = np.array(occl[:len(G)], dtype=bool)
    print(f"\n  === LA PORTE ({len(G)} pas) ===")
    print(f"    poids moyen  agentview {G[:,0].mean():.3f}   poignet {G[:,1].mean():.3f}")
    print(f"    écart-type   agentview {G[:,0].std():.3f}   poignet {G[:,1].std():.3f}")
    print(f"    min/max agentview : {G[:,0].min():.3f} / {G[:,0].max():.3f}")
    print(f"\n  === FACE À L'OCCLUSION RÉELLE ({O.mean()*100:.0f} % des pas) ===")
    if O.any() and (~O).any():
        print(f"    agentview VISIBLE  -> poids agentview {G[~O,0].mean():.3f}")
        print(f"    agentview MASQUÉE  -> poids agentview {G[O,0].mean():.3f}")
        d = G[~O,0].mean() - G[O,0].mean()
        print(f"    écart : {d:+.3f}  -> {'la porte RÉAGIT à l occlusion' if abs(d)>0.02 else 'la porte IGNORE l occlusion'}")
    else:
        print("    (pas assez d'occlusion pour comparer)")


if __name__ == "__main__":
    main()
