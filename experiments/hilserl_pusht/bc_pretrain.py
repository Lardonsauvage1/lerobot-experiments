#!/usr/bin/env python
"""BC pretrain de l'acteur SAC sur les démos Push-T (imitation supervisée, SANS reward).
-> produit un checkpoint pour warm-starter HIL-SERL (policy.pretrained_path).
Acteur sort du tanh [-1,1] ; action experte [0,512] normalisée -> [-1,1] ; loss = MSE(tanh(means), a_norm).
Entraîne l'encodeur + le réseau acteur + mean_layer (forward manuel sans detach)."""
import os
import draccus
import torch
import torch.nn.functional as F
import lerobot.policies.sac.configuration_sac  # noqa: F401
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.factory import make_dataset
from lerobot.rl.buffer import ReplayBuffer

CFG = "experiments/hilserl_pusht/pusht_hilserl.json"
OUT = "results/runs/hilserl_pusht/bc_actor"
EPOCHS = int(os.environ.get("BC_EPOCHS", "40"))
STEPS = int(os.environ.get("BC_STEPS", "200"))
SWA_LAST = int(os.environ.get("BC_SWA", "10"))   # moyenne des N dernières epochs (au lieu d'un cooldown)
BATCH = 128
LR = float(os.environ.get("BC_LR", "1e-4"))   # 3e-4 divergeait (NaN) sur l'encodeur CNN from-scratch
GRAD_CLIP = 1.0
AMIN, AMAX = 0.0, 512.0


def main():
    cfg = draccus.parse(TrainRLServerPipelineConfig, args=[f"--config_path={CFG}"])
    pol = make_policy(cfg.policy, env_cfg=cfg.env)
    pol.train()
    dev = next(pol.parameters()).device
    ds = make_dataset(cfg)
    rb = ReplayBuffer.from_lerobot_dataset(
        ds, state_keys=list(cfg.policy.input_features.keys()), device="cpu",
        optimize_memory=True, capacity=cfg.policy.offline_buffer_capacity,
    )
    print(f"[bc] buffer={len(rb)} device={dev} epochs={EPOCHS} steps/epoch={STEPS}", flush=True)
    opt = torch.optim.Adam(pol.actor.parameters(), lr=LR)
    swa_sum, swa_n = None, 0
    for ep in range(EPOCHS):
        tot = 0.0
        for _ in range(STEPS):
            b = rb.sample(BATCH)
            st = {k: v.to(dev) for k, v in b["state"].items()}
            a = b["action"].to(dev)
            a_norm = ((a - AMIN) / (AMAX - AMIN) * 2.0 - 1.0).clamp(-0.999, 0.999)
            enc = pol.actor.encoder(st)                 # pas de detach -> l'encodeur apprend
            out = pol.actor.network(enc)
            pred = torch.tanh(pol.actor.mean_layer(out))
            loss = F.mse_loss(pred, a_norm)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.actor.parameters(), GRAD_CLIP)   # évite l'explosion -> NaN
            opt.step()
            tot += float(loss.item())
        print(f"[bc] epoch {ep+1}/{EPOCHS}  bc_mse={tot/STEPS:.5f}", flush=True)
        # SWA : accumule les poids sur les N dernières epochs
        if ep >= EPOCHS - SWA_LAST:
            sd = pol.state_dict()
            if swa_sum is None:
                swa_sum = {k: v.detach().float().clone() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    swa_sum[k] += v.detach().float()
            swa_n += 1
    # applique la moyenne des derniers checkpoints (float uniquement ; garde les buffers int tels quels)
    if swa_sum is not None:
        cur = pol.state_dict()
        avg = {}
        for k, v in cur.items():
            avg[k] = (swa_sum[k] / swa_n).to(v.dtype) if torch.is_floating_point(v) else v
        pol.load_state_dict(avg)
        print(f"[bc] moyenne SWA des {swa_n} dernières epochs appliquée", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pol.save_pretrained(OUT)
    print(f"[bc] sauvé (SWA) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
