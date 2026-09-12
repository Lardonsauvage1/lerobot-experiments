#!/usr/bin/env python
"""Mesure la VRAIE latence d'inference d'un modele pomme (a lancer sur atomman, CPU).
- 1 inference reseau = 1 appel select_action qui declenche le calcul diffusion (100 pas) -> 16 actions.
- Les 7 appels suivants lisent la file (rapides). Le 8e re-declenche le reseau.
On chronometre : (a) 1 inference reseau complete, (b) un appel "file" (queue), (c) le debit soutenu.
"""
import sys, time
from pathlib import Path
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "deployable_models/apple/B_cd5k_from12000/brut")
NIS = int(sys.argv[2]) if len(sys.argv) > 2 else None   # num_inference_steps override (ex 16)
DEV = torch.device("cpu")
policy = DiffusionPolicy.from_pretrained(str(MODEL)); policy.eval().to(DEV)
if NIS:
    policy.config.num_inference_steps = NIS
policy.reset()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(MODEL),
    preprocessor_overrides={"device_processor": {"device": "cpu"}})
inp = policy.config.input_features
def obs():
    return {k: torch.rand(1, *tuple(v.shape), dtype=torch.float32) for k, v in inp.items()}

def step():
    with torch.no_grad():
        return post(policy.select_action(pre(obs())))

print(f"[bench] {MODEL} | num_inference_steps={policy.config.num_inference_steps} "
      f"horizon={policy.config.horizon} n_action_steps={policy.config.n_action_steps}")
step()  # warmup (1er = compile/alloc)
policy.reset()

# 16 appels d'affilee : le 1er et le 9e declenchent le reseau, les autres lisent la file
lat = []
for i in range(16):
    t = time.perf_counter(); step(); lat.append((time.perf_counter() - t) * 1000)
net = [lat[0], lat[8]]                       # inferences reseau
queue = [lat[i] for i in range(16) if i not in (0, 8)]
print(f"[bench] INFERENCE RESEAU (100 pas diffusion) : {sum(net)/len(net):.0f} ms  (echantillons {[round(x) for x in net]})")
print(f"[bench] appel FILE (queue)                  : {sum(queue)/len(queue):.1f} ms")
# debit soutenu : 16 pas = 2 inferences reseau + 14 lectures file
total_s = sum(lat) / 1000
print(f"[bench] 16 pas de controle en {total_s:.2f}s -> debit soutenu = {16/total_s:.1f} Hz (cible 15 Hz)")
print(f"[bench] budget par pas a 15 Hz = 66,7 ms ; 1 inference reseau doit tenir < 8*66,7 = 533 ms")
ok = (sum(net)/len(net)) < 533
print(f"[bench] TEMPS REEL 15 Hz {'OK' if ok else 'TROP LENT'} (inference reseau {'<' if ok else '>'} 533 ms)")
