"""Portfolio — filme la policy Lift et produit le GIF de la page d'accueil.

Filme N épisodes du val set (jamais entraînés) avec le harnais validé `lift_eval`,
garde le PREMIER épisode RÉUSSI, et le convertit en GIF dans docs/assets/.

Important : on filme la POLICY, pas les démonstrations expertes (cf. 55_demo_videos.py
qui rejoue les états enregistrés — ce serait malhonnête de l'étiqueter « le modèle »).

Usage : venv312/bin/python -u experiments/lift/91_portfolio_gif.py --ckpt <pretrained_model>
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

from src import lift_eval

HD = 256
NUM_INFERENCE_STEPS = 10
MAX_STEPS = 200
GIF_WIDTH = 300
GIF_FRAMES = 30


def to_gif(frames, dst, width=GIF_WIDTH, n=GIF_FRAMES):
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    imgs = []
    for i in idx:
        im = Image.fromarray(frames[i])
        h = int(im.height * width / im.width)
        imgs.append(im.resize((width, h), Image.LANCZOS)
                      .convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs[0].save(dst, save_all=True, append_images=imgs[1:],
                 duration=110, loop=0, optimize=True)
    return dst.stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out", default="docs/assets/demo_lift.gif")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = NUM_INFERENCE_STEPS
    pre = PolicyProcessorPipeline.from_pretrained(args.ckpt,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(args.ckpt,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    vid_dir = Path("results/runs/lift/90_portfolio_demo/videos")
    vid_dir.mkdir(parents=True, exist_ok=True)
    best = None
    n_ok = 0
    for ep in range(len(val_idx)):
        obs = lift_eval.fix_obs_sign(env.reset_to(dict(states=init_states[ep])))
        policy.reset()
        frames, success = [], False
        for _ in range(MAX_STEPS):
            frames.append(env.env.sim.render(height=HD, width=HD, camera_name="agentview")[::-1])
            img = env.env.sim.render(height=96, width=96, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            state_t = torch.from_numpy(lift_eval.build_state_vector(obs))
            obs_dict = pre({"observation.image": img_t.unsqueeze(0).to(device),
                            "observation.state": state_t.unsqueeze(0).to(device)})
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs = lift_eval.fix_obs_sign(env.step(a)[0])
            if env.is_success()["task"]:
                success = True
                break
        n_ok += success
        tag = "OK" if success else "FAIL"
        imageio.mimsave(vid_dir / f"portfolio_ep{val_idx[ep]}_{tag}.mp4", frames, fps=20)
        print(f"  ep {val_idx[ep]} -> {tag} ({len(frames)} frames)", flush=True)
        # on garde le PREMIER succès, et le plus court (le plus net visuellement)
        if success and (best is None or len(frames) < len(best[1])):
            best = (val_idx[ep], frames)

    print(f"\n{n_ok}/{len(val_idx)} réussis", flush=True)
    if best is None:
        print("AUCUN succès — pas de GIF (ne pas maquiller un échec en réussite)")
        return 1
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mb = to_gif(best[1], dst)
    print(f"GIF : {dst} ({mb:.2f} Mo) depuis ep{best[0]} ({len(best[1])} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
