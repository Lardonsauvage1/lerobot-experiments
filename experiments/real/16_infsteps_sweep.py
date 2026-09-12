#!/usr/bin/env python
"""Combien de pas de debruitage sans perdre en perf ? (modele pomme B brut)
Mesure l'ecart de l'ACTION produite quand on reduit num_inference_steps, vs la reference 100 pas.
- DDPM (defaut, entraine ainsi) 100 pas = reference.
- DDIM avec N pas (5..50) : conditionnement + bruit initial FIXES (seed) -> seul N varie.
Proxy offline de 'meme comportement' : |delta action| en rad (pas de simu pomme -> juge final = robot).
"""
import time, copy
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = "results/runs/real/apple_joint_224_r34/cooldown_012000/checkpoints/005000/pretrained_model"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

policy = DiffusionPolicy.from_pretrained(MODEL); policy.eval().to(DEV)
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": DEV.type}})
inp = policy.config.input_features

# observation FIXE (dummy mais identique pour toutes les configs)
torch.manual_seed(0)
OBS = {k: torch.rand(1, *tuple(v.shape), dtype=torch.float32) for k, v in inp.items()}
ddpm = policy.diffusion.noise_scheduler            # scheduler d'origine (DDPM)
ddim = DDIMScheduler(num_train_timesteps=ddpm.config.num_train_timesteps,
                     beta_start=ddpm.config.beta_start, beta_end=ddpm.config.beta_end,
                     beta_schedule=ddpm.config.beta_schedule, prediction_type=ddpm.config.prediction_type,
                     clip_sample=ddpm.config.clip_sample, clip_sample_range=ddpm.config.clip_sample_range)

def action_for(scheduler, n):
    policy.diffusion.noise_scheduler = scheduler
    policy.diffusion.num_inference_steps = n
    policy.reset(); torch.manual_seed(123)          # meme bruit initial pour tous
    with torch.no_grad():
        a = post(policy.select_action(pre({k: v.clone() for k, v in OBS.items()})))
    return a.squeeze(0).cpu().numpy()

def timed(scheduler, n, reps=3):
    policy.diffusion.noise_scheduler = scheduler; policy.diffusion.num_inference_steps = n
    ts = []
    for _ in range(reps):
        policy.reset(); torch.manual_seed(123); t = time.perf_counter()
        with torch.no_grad(): post(policy.select_action(pre({k: v.clone() for k, v in OBS.items()})))
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts)

ref = action_for(ddpm, 100)                          # reference DDPM 100 pas
t_ref = timed(ddpm, 100)
print(f"REF DDPM 100 pas : action={np.round(ref,4)}  ({t_ref:.0f} ms)")
print(f"\n{'sampler':10} {'N':>4} {'max|Δ| (rad)':>14} {'moy|Δ|':>10} {'latence ms':>11}")
print(f"{'DDPM':10} {100:>4} {0.0:>14.4f} {0.0:>10.4f} {t_ref:>11.0f}   <- reference")
for n in [50, 25, 16, 10, 8, 5, 3, 2]:
    a = action_for(ddim, n); d = np.abs(a - ref)
    print(f"{'DDIM':10} {n:>4} {d.max():>14.4f} {d.mean():>10.4f} {timed(ddim,n):>11.0f}")
# temoin : DDPM reduit (pour montrer que DDPM tient mal a bas N)
for n in [10, 5]:
    a = action_for(ddpm, n); d = np.abs(a - ref)
    print(f"{'DDPM':10} {n:>4} {d.max():>14.4f} {d.mean():>10.4f} {timed(ddpm,n):>11.0f}")
print("\nGuide : max|Δ| < ~0,01 rad (~0,6°) = action quasi identique. Juge final = robot.")
