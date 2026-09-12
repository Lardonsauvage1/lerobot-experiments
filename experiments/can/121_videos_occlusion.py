#!/usr/bin/env python
"""Vidéos du modèle témoin (entraîné sur du CLAIR) mis à l'épreuve de l'occlusion.

Chaque image est un montage :
    GAUCHE  = la scène telle qu'un humain la voit (256 px)
    DROITE  = CE QUE LE MODÈLE REÇOIT (96 px agrandi) — canette retirée quand le bras la masque
Sans le panneau de droite on ne comprend pas le comportement : on croit voir un robot qui
rate une cible visible, alors qu'il agit à l'aveugle.

Un bandeau rouge signale les instants d'occlusion et compte les cycles descente/remontée.

Épisodes choisis d'après les trajectoires déjà enregistrées : les échecs les plus
« boucleurs » (ép. 118 = 14 cycles) et deux réussites pour comparaison.

Sortie : results/runs/can/occluded/videos/*.mp4
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import importlib.util
from pathlib import Path

import imageio
import numpy as np
import torch

from src import can_eval, can_occlusion

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vis500)

CKPT = "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/occluded/videos"); OUT.mkdir(parents=True, exist_ok=True)
BIG, SMALL = 256, 96


def band(img, color, h=6):
    img = img.copy(); img[:h, :] = color; return img


def upscale_to(img, size):
    """96 -> 256 : le facteur n'est pas entier (256//96 = 2 donnait 192 px), donc PIL en
    NEAREST — on veut voir les vrais pixels que reçoit le réseau, pas une version lissée."""
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=[118, 42, 4])
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=300)
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    policy = DiffusionPolicy.from_pretrained(CKPT).to(device).eval()
    policy.diffusion.num_inference_steps = 10
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    env = can_eval.make_env()
    for ep in a.episodes:
        obs = env.reset_to(dict(states=states[ep]))
        policy.reset()
        frames, zs, n_occ = [], [], 0
        success = False
        for _ in range(a.max_steps):
            hide = can_occlusion.can_is_occluded(obs, a.radius)
            n_occ += int(hide)
            big = can_occlusion.render_normal(env, BIG, BIG)
            small = (can_occlusion.render_without_can(env, SMALL, SMALL) if hide
                     else can_occlusion.render_normal(env, SMALL, SMALL))
            # rouge quand la cible est masquée, vert sinon — sur le panneau du MODÈLE
            view = band(upscale_to(small, BIG), (220, 40, 40) if hide else (40, 160, 40))
            frames.append(np.concatenate([big, view], axis=1))

            img_t = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            st = torch.from_numpy(vis500.state_proprio(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = policy.select_action(pre({"observation.image": img_t,
                                                "observation.state": st}))
            act = np.clip(post(act).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            zs.append(float(np.asarray(obs["robot0_eef_pos"]).flatten()[2]))
            obs = env.step(act)[0]
            if env.is_success()["task"]:
                success = True; break

        cyc = can_occlusion.z_cycles(zs)
        tag = "OK" if success else "ECHEC"
        name = f"ep{ep:03d}_{tag}_{cyc}cycles_r{int(a.radius*100):02d}.mp4"
        imageio.mimsave(OUT / name, frames, fps=20)
        print(f"  ép {ep:3d} : {tag:5s} | {cyc:2d} cycles | {n_occ:3d}/{len(frames)} pas occlus "
              f"({n_occ/max(len(frames),1):.0%}) -> {name}", flush=True)
    print(f"\n✓ {OUT}   (gauche = vue humaine · droite = ce que le modèle reçoit, "
          f"bandeau ROUGE = cible masquée)")


if __name__ == "__main__":
    main()
