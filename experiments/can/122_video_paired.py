#!/usr/bin/env python
"""LA vidéo qui isole l'effet de l'occlusion : MÊME épisode, joué dans les DEUX conditions.

Pourquoi celle-ci et pas les précédentes. Filmer « un échec sous occlusion » ne prouve rien :
l'occlusion n'est ni nécessaire ni suffisante pour rater (ép. 118 échoue avec 0 % d'occlusion,
ép. 4 réussit avec 58 %). Sur un cas isolé, on ne peut rien conclure.

La seule mise en scène qui isole l'effet est APPARIÉE : même état initial, même modèle,
même seed — seule l'occlusion diffère.

    GAUCHE  = sans occlusion (le témoin réussit)
    DROITE  = avec occlusion à 3 cm (bandeau rouge quand la cible est masquée)

Les épisodes sont pris parmi les 89 DISCORDANTS identifiés par l'analyse appariée : réussis
au témoin, ratés à 3 cm après avoir réellement subi une occlusion. Ce sont eux qui portent
l'effet mesuré (94 perdus contre 21 gagnés, p = 3e-12).

⚠️ Les rollouts ne sont pas déterministes (la diffusion échantillonne) : un épisode
discordant dans les données peut ne pas l'être au rejeu. Le script en essaie plusieurs et
ne garde que ceux qui reproduisent le contraste.

Sortie : results/runs/can/occluded/videos/paired_*.mp4
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
from PIL import Image

from src import can_eval, can_occlusion

spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vis500)

CKPT = "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model"
OUT = Path("results/runs/can/occluded/videos"); OUT.mkdir(parents=True, exist_ok=True)
SIZE, MAXS = 256, 300


def rollout(env, policy, pre, post, state, radius, device):
    """Retourne (frames, succès). Chaque frame = vue 256 px, bandeau rouge si cible masquée."""
    obs = env.reset_to(dict(states=state))
    policy.reset()
    frames, succ = [], False
    for _ in range(MAXS):
        hide = can_occlusion.can_is_occluded(obs, radius)
        big = can_occlusion.render_normal(env, SIZE, SIZE)
        big = big.copy(); big[:8, :] = (220, 40, 40) if hide else (40, 160, 40)
        frames.append(big)
        small = (can_occlusion.render_without_can(env, 96, 96) if hide
                 else can_occlusion.render_normal(env, 96, 96))
        img = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        st = torch.from_numpy(vis500.state_proprio(obs)).unsqueeze(0).to(device)
        with torch.no_grad():
            a = policy.select_action(pre({"observation.image": img, "observation.state": st}))
        a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
        obs = env.step(a)[0]
        if env.is_success()["task"]:
            succ = True; break
    return frames, succ


def label(img, txt, ok):
    from PIL import ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, img.shape[0] - 18, img.shape[1], img.shape[0]], fill=(0, 0, 0))
    d.text((5, img.shape[0] - 15), txt, fill=(90, 255, 90) if ok else (255, 90, 90))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2, 6, 14, 18])
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--keep", type=int, default=3, help="nb de vidéos contrastées à garder")
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
    kept = 0
    for ep in a.episodes:
        if kept >= a.keep:
            break
        fr_clear, ok_clear = rollout(env, policy, pre, post, states[ep], 0.0, device)
        fr_occ, ok_occ = rollout(env, policy, pre, post, states[ep], a.radius, device)
        contrast = ok_clear and not ok_occ
        print(f"  ép {ep:3d} : sans occlusion {'OK ' if ok_clear else 'ÉCHEC'} | "
              f"avec occlusion {'OK ' if ok_occ else 'ÉCHEC'} -> "
              f"{'CONTRASTE ✔' if contrast else 'pas de contraste au rejeu'}", flush=True)
        if not contrast:
            continue
        n = max(len(fr_clear), len(fr_occ))
        pad = lambda f: f + [f[-1]] * (n - len(f))
        frames = [np.concatenate([label(l, "sans occlusion", ok_clear),
                                  label(r, f"occlusion {a.radius*100:.0f} cm", ok_occ)], axis=1)
                  for l, r in zip(pad(fr_clear), pad(fr_occ))]
        name = f"paired_ep{ep:03d}_r{int(a.radius*100):02d}.mp4"
        imageio.mimsave(OUT / name, frames, fps=20)
        print(f"      -> {name}", flush=True)
        kept += 1
    print(f"\n✓ {kept} vidéo(s) appariée(s) dans {OUT}")


if __name__ == "__main__":
    main()
