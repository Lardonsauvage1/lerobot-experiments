#!/usr/bin/env python
"""Benchmark de la VITESSE DE CALCUL d'un pas d'entraînement (forward+backward+step), batch synthétique.
Portable : détecte cuda/mps/cpu. Compare les machines à archi/config égales, sans le dataloader.
Usage: python 22_bench_step.py <chemin_checkpoint> [batch]"""
import sys, time, torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CKPT = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if DEV.type == "cpu":
    torch.set_num_threads(12)

p = DiffusionPolicy.from_pretrained(CKPT).to(DEV).train()
c = p.config; nobs, H = c.n_obs_steps, c.horizon
sdim = c.input_features["observation.state"].shape[0]
adim = c.output_features["action"].shape[0]
def synth():
    b = {k: torch.rand(B, nobs, 3, 224, 224, device=DEV) for k in c.image_features}
    b["observation.state"] = torch.rand(B, nobs, sdim, device=DEV)
    b["action"] = torch.rand(B, H, adim, device=DEV)
    b["action_is_pad"] = torch.zeros(B, H, dtype=torch.bool, device=DEV)
    return b
opt = torch.optim.Adam(p.parameters(), lr=1e-4)
def sync():
    if DEV.type == "cuda": torch.cuda.synchronize()

N = 30
for i in range(N + 5):                       # 5 de warmup
    if i == 5: sync(); t0 = time.perf_counter()
    loss = p.forward(synth())[0]
    opt.zero_grad(); loss.backward(); opt.step()
sync(); dt = (time.perf_counter() - t0) / N
print(f"device={DEV.type} batch={B} : {dt*1000:.0f} ms/step  ->  {dt:.3f} s/step  ({B/dt:.1f} échantillons/s)")
