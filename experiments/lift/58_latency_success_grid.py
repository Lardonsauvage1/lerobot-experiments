"""Phase 4 — grille latence × succès : lignes = pas de diffusion, colonnes = taille U-Net.

Pour chaque case (taille, pas) : mesure la latence d'un sample (obs factices) ET le succès
(rollout 50 init states val figés). Sauvegarde incrémentale après CHAQUE case (survit aux
veilles / interruptions). Rend un tableau markdown + JSON.

Sortie : results/runs/lift/51_unet_sweep_eval/grid.{json,md}
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import time
from pathlib import Path

import numpy as np
import torch

from src import lift_eval

OUT = Path("results/runs/lift/51_unet_sweep_eval")
STEPS_LIST = [10, 5, 4, 2, 1]
N_WARMUP, N_LAT_TRIALS = 2, 6
# (label, checkpoint au meilleur step) — baseline + 6 tailles du sweep
SIZES = [
    ("[512,1024,2048]", "results/runs/lift/47_baseline_150/checkpoints/006000/pretrained_model"),
    ("[256,512,1024]", "results/runs/lift/51_unet_d256_512_1024/checkpoints/004500/pretrained_model"),
    ("[128,256,512]", "results/runs/lift/51_unet_d128_256_512/checkpoints/004500/pretrained_model"),
    ("[64,128,256]", "results/runs/lift/51_unet_d64_128_256/checkpoints/006000/pretrained_model"),
    ("[32,64,128]", "results/runs/lift/51_unet_d32_64_128/checkpoints/007500/pretrained_model"),
    ("[16,32,64]", "results/runs/lift/51_unet_d16_32_64/checkpoints/010500/pretrained_model"),
    ("[8,16,32]", "results/runs/lift/51_unet_d8_16_32/checkpoints/009000/pretrained_model"),
]


def fake_obs(device):
    return {"observation.image": torch.rand(1, 3, 96, 96, device=device),
            "observation.state": torch.rand(1, 19, device=device)}


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def measure_latency(policy, device, steps):
    policy.diffusion.num_inference_steps = steps
    with torch.no_grad():
        for _ in range(N_WARMUP):
            policy.reset(); _ = policy.select_action(fake_obs(device)); sync(device)
        ts = []
        for _ in range(N_LAT_TRIALS):
            policy.reset()
            sync(device); t0 = time.perf_counter()
            _ = policy.select_action(fake_obs(device))
            sync(device); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    split = lift_eval.load_or_make_split()
    val_idx = split["val"]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    gj = OUT / "grid.json"
    cells = json.loads(gj.read_text())["cells"] if gj.exists() else []
    done = {(c["size"], c["steps"]) for c in cells}

    for label, ckpt in SIZES:
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        params = sum(p.numel() for p in policy.parameters())
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        print(f"\n##### {label} ({params/1e6:.1f}M) #####", flush=True)
        for steps in STEPS_LIST:
            if (label, steps) in done:
                print(f"  steps={steps}: déjà fait, skip", flush=True); continue
            lat = measure_latency(policy, device, steps)
            _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
                device=device, num_inference_steps=steps, verbose=False)
            cells.append({"size": label, "params": params, "steps": steps, "latency_ms": lat,
                          "success_rate": agg["success_rate"], "t_success_median": agg["t_success_median"],
                          "max_z_mean": agg["max_z_mean"]})
            gj.write_text(json.dumps({"n": len(val_idx), "cells": cells}, indent=2))
            print(f"  steps={steps:2d} : {lat:6.1f} ms | succès {agg['success_rate']:.0%}", flush=True)
        del policy, pre, post

    render_table(cells)


def render_table(cells):
    sizes, steps = [], []
    for c in cells:
        if c["size"] not in sizes: sizes.append(c["size"])
        if c["steps"] not in steps: steps.append(c["steps"])
    # colonnes triées par params décroissant
    pmap = {c["size"]: c["params"] for c in cells}
    sizes = sorted(sizes, key=lambda s: -pmap[s])
    steps = sorted(steps, reverse=True)
    cell = {(c["size"], c["steps"]): c for c in cells}

    lines = ["# Grille latence × succès — [32,64,128] @4 pas = le sweet spot",
             "", "Cases : **latence d'un sample (ms) · succès (50 val ép.)**. Lignes = pas de diffusion, colonnes = U-Net.", ""]
    hdr = "| pas \\ U-Net | " + " | ".join(f"{s}<br>{pmap[s]/1e6:.0f}M" for s in sizes) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(sizes) + 1))
    for st in steps:
        row = [f"**{st}**"]
        for s in sizes:
            c = cell.get((s, st))
            row.append(f"{c['latency_ms']:.0f}ms · {c['success_rate']:.0%}" if c else "—")
        lines.append("| " + " | ".join(row) + " |")
    (OUT / "grid.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nJSON: {OUT/'grid.json'}\nTableau: {OUT/'grid.md'}")


if __name__ == "__main__":
    main()
