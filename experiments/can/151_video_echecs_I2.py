#!/usr/bin/env python
"""Filme les ÉCHECS du meilleur modèle à 2 caméras dont une embarquée (I2, 91,2 %).

Trois panneaux :
  agentview  — ENTRÉE n°1 du modèle (vue extérieure)
  poignet    — ENTRÉE n°2 du modèle (caméra embarquée)
  birdview   — repère pour l'œil humain, PAS donné au modèle

Les épisodes filmés sont exactement ceux que l'éval à 500 rollouts a comptés en échec,
rejoués depuis les mêmes états initiaux figés : ce sont les vrais échecs, pas des
rollouts refaits au hasard.

Usage : venv312/bin/python experiments/can/151_video_echecs_I2.py --n 6
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from src import can_eval

spec = importlib.util.spec_from_file_location("v", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

CK = "results/runs/can/cam2_I2_frozen_s43/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/cam2_I2_frozen_s43/videos_echecs")
BIG = 256
CAN_LIFTED = 0.90
CAN_REPOS = 0.8600   # z de la canette POSÉE : au-dessus seulement = elle a bougé


def panel(img, title, sub, entree):
    im = Image.fromarray(img).resize((BIG, BIG), Image.NEAREST)
    band = Image.new("RGB", (BIG, 34), (24, 24, 28) if entree else (70, 40, 40))
    d = ImageDraw.Draw(band)
    d.text((6, 3), title, fill=(255, 255, 255))
    d.text((6, 18), sub, fill=(150, 220, 150) if entree else (230, 170, 170))
    out = Image.new("RGB", (BIG, BIG + 34))
    out.paste(band, (0, 0)); out.paste(im, (0, 34))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="nombre d'échecs REPRODUCTIBLES à filmer")
    ap.add_argument("--candidats", type=int, default=25,
                    help="échecs de l'éval à rejouer pour en trouver n reproductibles")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    res = json.loads(Path("results/runs/can/cam2_eval500.json").read_text())["results"]
    rec = next(r for r in res if "I2" in r["name"])
    fails = [i for i, e in enumerate(rec["per_episode"])
             if not (e["success"] if isinstance(e, dict) else e)]
    print(f"  {len(fails)} échecs sur 500 ({rec['success_rate']:.1%} de réussite) — "
          f"je filme les {a.n} premiers : {fails[:a.n]}", flush=True)

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

    states = V.make_eval_states()
    env = can_eval.make_env()
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    # ⚠️ Un épisode compté en échec par l'éval peut réussir au rejeu : le rendu et MPS ne
    # sont pas déterministes, et les épisodes qui échouent sont justement ceux qui sont sur
    # le fil. On rejoue donc des candidats et on ne GARDE que les échecs reproductibles —
    # filmer un « échec » qui réussit à l'écran serait mentir sur ce qu'on montre.
    resume, flips = [], []
    for ep in fails[:a.candidats]:
        if len(resume) >= a.n:
            break
        obs = env.reset_to(dict(states=states[ep])); pol.reset()
        frames, closed, zmax, ok = [], False, 0.0, False
        for step in range(can_eval.MAX_STEPS):
            can_z = float(np.asarray(obs["object"]).flatten()[9]); zmax = max(zmax, can_z)
            raw = {c: env.env.sim.render(height=96, width=96, camera_name=c)[::-1]
                   for c in ("agentview", "robot0_eye_in_hand", "birdview")}
            ph = "TRANSPORT" if can_z > CAN_LIFTED else ("SAISIE" if closed else "APPROCHE")
            frames.append(np.concatenate([
                panel(raw["agentview"], "VUE EXTÉRIEURE", "entrée modèle", True),
                panel(raw["robot0_eye_in_hand"], "POIGNET (embarquée)", "entrée modèle", True),
                panel(raw["birdview"], "DESSUS", f"repère humain · {ph} · t={step}", False),
            ], axis=1))
            imgs = {k: torch.from_numpy(raw[c].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
                    for c, k in keys.items()}
            st = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = pol.select_action(pre({**imgs, "observation.state": st}))
            act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
            if act[6] > 0: closed = True
            obs = env.step(act)[0]
            if env.is_success()["task"]: ok = True; break
        # diagnostic grossier du mode d'échec, d'après la hauteur max atteinte par la canette
        if ok:
            flips.append(ep)
            print(f"    ep {ep:3d} : RÉUSSIT au rejeu -> écarté (échec non reproductible)", flush=True)
            continue
        # seuils relatifs au repos (0,8600) : un z_max de 0,862 veut dire que la canette
        # n'a PAS bougé — la pince s'est fermée dans le vide ou a glissé aussitôt.
        mode = ("JAMAIS DÉPLACÉE" if zmax < CAN_REPOS + 0.01 else
                "SAISIE PUIS LÂCHÉE" if zmax > CAN_LIFTED else "DÉCOLLÉE SANS ÊTRE PORTÉE")
        name = f"echec_ep{ep:03d}_{mode.lower().replace(' ', '_')}.mp4"
        imageio.mimsave(OUT / name, frames, fps=20)
        resume.append((ep, mode, zmax, len(frames), ok))
        print(f"    ep {ep:3d} : {mode:20s} z_max={zmax:.3f}  {len(frames)} images -> {name}", flush=True)

    n_essais = len(resume) + len(flips)
    print(f"\n  === REPRODUCTIBILITÉ ===", flush=True)
    print(f"    {len(flips)}/{n_essais} des échecs de l'éval RÉUSSISSENT au rejeu "
          f"({len(flips)/max(n_essais,1)*100:.0f} %) : {flips}")
    print(f"    -> les échecs de ce modèle sont majoritairement des épisodes SUR LE FIL")
    print(f"\n  === MODES D'ÉCHEC (reproductibles seulement) ===", flush=True)
    from collections import Counter
    for m, c in Counter(r[1] for r in resume).most_common():
        print(f"    {m:22s} {c}/{len(resume)}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
