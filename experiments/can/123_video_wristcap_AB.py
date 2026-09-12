#!/usr/bin/env python
"""La vidéo qui montre pourquoi il n'y a PAS de caméra de poignet sur le modèle final.

Deux modèles ENTRAÎNÉS À L'IDENTIQUE (ResNet34, U-Net [128,256,512], LR const + EMA,
30k steps, cooldown 5k, mêmes démonstrations). Une seule différence : B reçoit en plus
le flux de la caméra embarquée dans la pince, avec son propre encodeur.

    A (agentview seule)      67,4 % sur 500 rollouts  [63,2-71,4]
    B (agentview + poignet)  42,0 % sur 500 rollouts  [37,8-46,4]

Soit −25 points EN AJOUTANT de l'information. Le résultat tient à trois capacités et
deux résolutions (96/224/84 px) : ce n'est pas un accident d'entraînement.

GAUCHE = A. DROITE = B, avec en médaillon ce que voit sa caméra de poignet.
On ne garde que les épisodes DISCORDANTS (A réussit, B échoue) : eux seuls portent l'effet.

Sortie : results/runs/can/wristcap_AB/videos/
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
from PIL import Image, ImageDraw

from src import can_eval

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vis500)

CKPT_A = "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model"
CKPT_B = "results/runs/can/wristcap_B_wrist/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/wristcap_AB/videos")
SIZE, SMALL, MAXS = 256, 96, 300


def load(ckpt, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    p = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    p.diffusion.num_inference_steps = 10
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return p, pre, post


def render(env, cam, h, w):
    return env.env.sim.render(height=h, width=w, camera_name=cam)[::-1]


def rollout(env, policy, pre, post, state, device, *, with_wrist):
    """with_wrist=False -> bras A (1 caméra). True -> bras B (2 caméras, encodeurs séparés)."""
    obs = env.reset_to(dict(states=state))
    policy.reset()
    frames, wrists, succ = [], [], False
    for _ in range(MAXS):
        frames.append(render(env, "agentview", SIZE, SIZE))
        ag = render(env, "agentview", SMALL, SMALL)
        t_ag = torch.from_numpy(ag.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        st = torch.from_numpy(vis500.state_proprio(obs)).unsqueeze(0).to(device)
        if with_wrist:
            wr = render(env, "robot0_eye_in_hand", SMALL, SMALL)
            wrists.append(wr)
            t_wr = torch.from_numpy(wr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            batch = {"observation.images.agentview": t_ag,
                     "observation.images.wrist": t_wr, "observation.state": st}
        else:
            batch = {"observation.image": t_ag, "observation.state": st}
        with torch.no_grad():
            a = policy.select_action(pre(batch))
        a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
        obs = env.step(a)[0]
        if env.is_success()["task"]:
            succ = True; break
    return frames, wrists, succ


def panel(img, txt, ok, inset=None):
    im = Image.fromarray(img.copy())
    if inset is not None:
        small = Image.fromarray(inset).resize((72, 72), Image.LANCZOS)
        im.paste(small, (im.width - 78, 6))
        ImageDraw.Draw(im).rectangle([im.width - 79, 5, im.width - 6, 78], outline=(255, 210, 60))
    d = ImageDraw.Draw(im)
    d.rectangle([0, im.height - 18, im.width, im.height], fill=(0, 0, 0))
    d.text((5, im.height - 15), txt, fill=(90, 255, 90) if ok else (255, 90, 90))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*",
                    default=list(range(24)), help="indices dans can_eval500.npy")
    ap.add_argument("--keep", type=int, default=2)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    states = vis500.make_eval_states()
    pa = load(CKPT_A, device)
    pb = load(CKPT_B, device)
    env = can_eval.make_env()

    kept, tally = 0, {"A": 0, "B": 0, "n": 0}
    for ep in a.episodes:
        if kept >= a.keep:
            break
        fa, _, ok_a = rollout(env, *pa, states[ep], device, with_wrist=False)
        fb, wb, ok_b = rollout(env, *pb, states[ep], device, with_wrist=True)
        tally["n"] += 1; tally["A"] += ok_a; tally["B"] += ok_b
        print(f"  ép {ep:3d} : A {'OK ' if ok_a else 'ÉCHEC'} | B {'OK ' if ok_b else 'ÉCHEC'}"
              f"   (cumul A {tally['A']}/{tally['n']} · B {tally['B']}/{tally['n']})", flush=True)
        if not (ok_a and not ok_b):
            continue
        n = max(len(fa), len(fb))
        pad = lambda f: list(f) + [f[-1]] * (n - len(f))
        wb_p = pad(wb) if wb else [None] * n
        frames = [np.concatenate([
            panel(l, "A : agentview seule", ok_a),
            panel(r, "B : + camera poignet", ok_b, inset=w)], axis=1)
            for l, r, w in zip(pad(fa), pad(fb), wb_p)]
        name = f"wristAB_ep{ep:03d}.mp4"
        imageio.mimsave(OUT / name, frames, fps=20)
        print(f"      -> {name} ({n} frames)", flush=True)
        kept += 1

    print(f"\n{kept} vidéo(s) discordante(s) dans {OUT}")
    print(f"sur {tally['n']} épisodes rejoués : A {tally['A']}, B {tally['B']}")


if __name__ == "__main__":
    main()
