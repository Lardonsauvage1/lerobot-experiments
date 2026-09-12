#!/usr/bin/env python
"""Filme le défaut de PRÉHENSION : la pince remonte pendant qu'elle se ferme.

Mesuré sur 150 épisodes (oracle, position exacte connue) : au moment de la commande de
fermeture les deux groupes descendent encore, mais JUSTE APRÈS les échecs remontent dans
53 % des cas contre 16 % pour les réussites. La fermeture des doigts prend du temps ; si le
bras repart vers le haut pendant ce temps, la pince se referme au-dessus de l'objet.

Chaque image porte : hauteur de la pince au-dessus de la canette, état de la commande de
pince, et sens du mouvement vertical. Bandeau ROUGE dès que la pince se ferme EN MONTANT.

Sortie : results/runs/can/occluded/videos/prehension_*.mp4
"""
import sys
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
from pathlib import Path
import imageio, numpy as np, torch
from PIL import Image, ImageDraw

from src import can_eval, can_occlusion

import importlib.util
spec = importlib.util.spec_from_file_location("vis500", "experiments/can/10_vision_500_rollouts.py")
vis500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vis500)

OUT = Path("results/runs/can/occluded/videos"); OUT.mkdir(parents=True, exist_ok=True)
BIG = 320

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/runs/can/atomman/oracle")
ap.add_argument("--radius", type=float, default=0.03)
ap.add_argument("--episodes", type=int, nargs="*", default=list(range(0, 40)))
ap.add_argument("--keep", type=int, default=3)
a = ap.parse_args()

dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
states = vis500.make_eval_states()
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
    policy_action_to_transition, transition_to_policy_action)
pol = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval()
pol.diffusion.num_inference_steps = 10
pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
    to_transition=batch_to_transition, to_output=transition_to_batch)
post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
oracle = int(pol.config.input_features["observation.state"].shape[0]) == 12
render_fn = can_occlusion.make_render_fn(a.radius)

env = can_eval.make_env(); kept = 0
for ep in a.episodes:
    if kept >= a.keep: break
    obs = env.reset_to(dict(states=states[ep])); pol.reset()
    frames, zs, grip, dzs, canz, succ = [], [], [], [], [], False
    for t in range(can_eval.MAX_STEPS):
        cp = np.asarray(obs["object"]).flatten()[7:10]
        eef = np.asarray(obs["robot0_eef_pos"]).flatten()
        img = can_occlusion.render_normal(env, BIG, BIG)
        small = render_fn(env, obs, 96)
        pro = vis500.state_proprio(obs)
        if oracle: pro = np.concatenate([pro, cp]).astype(np.float32)
        o = pre({"observation.image": torch.from_numpy(small).permute(2,0,1).float().unsqueeze(0).to(dev)/255.,
                 "observation.state": torch.tensor(pro, device=dev).unsqueeze(0)})
        with torch.no_grad(): act = pol.select_action(o)
        act = np.clip(post(act).squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)
        zs.append(float(eef[2])); grip.append(float(act[6])); dzs.append(float(eef[2]-cp[2]))
        canz.append(float(cp[2]))
        vz = zs[-1]-zs[-4] if len(zs) > 3 else 0.0
        ferme_en_montant = act[6] > 0 and vz > 0.001
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        if ferme_en_montant: d.rectangle([0,0,BIG,10], fill=(230,40,40))
        d.rectangle([0,BIG-34,BIG,BIG], fill=(0,0,0))
        d.text((6,BIG-30), f"hauteur/canette {dzs[-1]*100:+5.1f} cm", fill=(255,255,255))
        d.text((6,BIG-17), f"pince {'FERME' if act[6]>0 else 'ouverte'}   "
                           f"vertical {'MONTE' if vz>0.001 else ('descend' if vz<-0.001 else 'stable')}",
               fill=(255,90,90) if ferme_en_montant else (180,255,180))
        frames.append(np.asarray(im))
        obs = env.step(act)[0]
        if env.is_success()["task"]: succ = True; break
    g = np.array(grip); z = np.array(zs); dz = np.array(dzs)
    idx = np.where(np.diff((g > 0).astype(int)) == 1)[0]
    # ⚠️ CORRECTION : ne garder QUE les vrais échecs de PRÉHENSION — la canette ne doit
    # JAMAIS avoir décollé. Sans ce filtre on filme des saisies réussies suivies d'un dépôt
    # au mauvais compartiment (le bac en a 4), ce qui est un autre mode d'échec.
    souleve = max(canz) > can_occlusion.CAN_Z_REST + can_occlusion.LIFT_MARGIN
    if succ or souleve or not len(idx):
        raison = "réussi" if succ else ("SAISIE OK puis mauvais dépôt" if souleve else "pas de fermeture")
        print(f"  ép {ep:3d} : {raison} — ignoré", flush=True); continue
    k = idx[0]+1
    if k+4 >= len(z): continue
    monte_apres = (z[k+3]-z[k]) > 0
    print(f"  ép {ep:3d} : ÉCHEC DE PRÉHENSION (jamais soulevée) | ferme à {dz[k]*100:.1f} cm | "
          f"{'remonte' if monte_apres else 'reste bas'}", flush=True)
    name = f"prehension_ep{ep:03d}_ferme{dz[k]*100:.0f}cm.mp4"
    imageio.mimsave(OUT/name, frames, fps=20)
    print(f"      -> {name}", flush=True)
    kept += 1
print(f"\n✓ {kept} vidéo(s) dans {OUT}")
