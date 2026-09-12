#!/usr/bin/env python3
"""SWA : moyenne les POIDS de plusieurs checkpoints joint et sauve un pretrained_model
évaluable (à <out>/checkpoints/000000/pretrained_model, donc eval avec --steps 0).
Les tenseurs flottants sont moyennés ; les non-flottants (et buffers de normalisation,
identiques entre checkpoints) sont repris du 1er.
  venv312/bin/python experiments/can/65_swa_make.py --steps 30000,24000,46000 --out /tmp/swa_best3
"""
import argparse, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--steps", required=True)
ap.add_argument("--src", default="results/runs/can/joint_r34_bigunet")
ap.add_argument("--out", required=True)
a = ap.parse_args()

steps = [int(x) for x in a.steps.split(",")]
srcs = [Path(a.src) / "checkpoints" / f"{s:06d}" / "pretrained_model" for s in steps]
for m in srcs:
    assert (m / "model.safetensors").exists(), f"manque {m}"

sds = [load_file(str(m / "model.safetensors")) for m in srcs]
avg = {}
for k, t0 in sds[0].items():
    if t0.is_floating_point():
        acc = torch.zeros_like(t0, dtype=torch.float64)
        for sd in sds:
            acc += sd[k].to(torch.float64)
        avg[k] = (acc / len(sds)).to(t0.dtype)
    else:
        avg[k] = t0.clone()

out_pm = Path(a.out) / "checkpoints" / "000000" / "pretrained_model"
if out_pm.exists():
    shutil.rmtree(out_pm)
out_pm.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(srcs[0], out_pm)
save_file(avg, str(out_pm / "model.safetensors"))
print(f"SWA: moyenné {len(steps)} checkpoints {steps} -> {out_pm}")
