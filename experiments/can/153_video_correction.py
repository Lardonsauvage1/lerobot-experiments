#!/usr/bin/env python
"""Filme la CORRECTION : on téléporte la canette en pleine approche, le bras se recale.

Preuve visuelle que la policy est en boucle fermée — la vision est rafraîchie jusqu'au
contact, la trajectoire n'est pas figée au départ.

Le bandeau du haut change au moment de la téléportation, et la vue de dessus affiche la
distance courante de la pince à la NOUVELLE et à l'ANCIENNE position de la canette :
on voit l'une fondre pendant que l'autre grandit.

Usage : venv312/bin/python experiments/can/153_video_correction.py --eps 12,29,38
"""
import argparse
import importlib.util
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
OUT = Path("results/runs/can/cam2_I2_frozen_s43/videos_correction")
BIG, DECLENCHE = 256, 0.15


def panel(img, title, sub, couleur):
    im = Image.fromarray(img).resize((BIG, BIG), Image.NEAREST)
    band = Image.new("RGB", (BIG, 34), couleur)
    d = ImageDraw.Draw(band)
    d.text((6, 3), title, fill=(255, 255, 255))
    d.text((6, 18), sub, fill=(225, 225, 225))
    out = Image.new("RGB", (BIG, BIG + 34))
    out.paste(band, (0, 0)); out.paste(im, (0, 34))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="12,29,38,19")
    ap.add_argument("--delta", type=float, default=0.04)
    a = ap.parse_args()
    eps = [int(x) for x in a.eps.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    pol = DiffusionPolicy.from_pretrained(CK).to(device).eval()
    pol.diffusion.num_inference_steps = V.STEPS
    pre = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = V.make_eval_states()
    env = can_eval.make_env(); sim = env.env.sim
    keys = {"agentview": "observation.images.agentview",
            "robot0_eye_in_hand": "observation.images.wrist"}
    for ep in eps:
        obs = env.reset_to(dict(states=states[ep])); pol.reset()
        can0 = np.asarray(obs["object"]).flatten()[7:10].copy()
        can_now, bouge, closed, ok = can0.copy(), False, False, False
        frames, t_tp = [], None
        for step in range(can_eval.MAX_STEPS):
            eef = np.asarray(obs["robot0_eef_pos"]).flatten()
            if (not bouge) and np.linalg.norm(eef[:2] - can_now[:2]) < DECLENCHE:
                q = sim.data.get_joint_qpos("Can_joint0").copy()
                ang = np.random.default_rng(ep).uniform(0, 2 * np.pi)
                q[0] += a.delta * np.cos(ang); q[1] += a.delta * np.sin(ang)
                sim.data.set_joint_qpos("Can_joint0", q); sim.forward()
                can_now = q[:3].copy(); bouge = True; t_tp = step
            raw = {c: env.env.sim.render(height=96, width=96, camera_name=c)[::-1]
                   for c in ("agentview", "robot0_eye_in_hand", "birdview")}
            dn = np.linalg.norm(eef[:2] - can_now[:2]) * 100
            do = np.linalg.norm(eef[:2] - can0[:2]) * 100
            if not bouge:
                tete, coul = "AVANT — canette à sa place", (24, 24, 28)
            elif step - t_tp < 15:
                tete, coul = f"CANETTE DÉPLACÉE DE {a.delta*100:.0f} cm", (150, 30, 30)
            else:
                tete, coul = "APRÈS — le bras se recale", (25, 90, 45)
            frames.append(np.concatenate([
                panel(raw["agentview"], "VUE EXTÉRIEURE", "entrée modèle", coul),
                panel(raw["robot0_eye_in_hand"], "POIGNET (embarquée)", "entrée modèle", coul),
                panel(raw["birdview"], tete,
                      f"nouvelle {dn:4.1f} cm · ancienne {do:4.1f} cm · t={step}", coul),
            ], axis=1))
            imgs = {k: torch.from_numpy(raw[c].copy()).permute(2, 0, 1).float()
                    .unsqueeze(0).to(device) / 255. for c, k in keys.items()}
            st = torch.from_numpy(V.state_proprio(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = pol.select_action(pre({**imgs, "observation.state": st}))
            act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
            if act[6] > 0 and not closed:
                closed = True; d_fin = (dn, do)
            obs = env.step(act)[0]
            if env.is_success()["task"]: ok = True; break
        for _ in range(20):
            frames.append(frames[-1])           # pause finale
        name = f"correction_ep{ep:03d}_{'reussi' if ok else 'rate'}.mp4"
        imageio.mimsave(OUT / name, frames, fps=20)
        print(f"    ep {ep:3d} : téléportée à t={t_tp}, ferme à {d_fin[0]:.1f} cm de la nouvelle "
              f"(vs {d_fin[1]:.1f} de l'ancienne) — {'RÉUSSI' if ok else 'raté'} -> {name}", flush=True)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
