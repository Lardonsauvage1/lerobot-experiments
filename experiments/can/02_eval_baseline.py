"""Phase 5 (Can) — éval du baseline : val-loss par checkpoint → best → rollout sur les 50 val.

But : confirmer si Can est résolu (et si l'archi compressée [64,128,256] mini-CNN généralise).
État Can 12D (proprio + can_pos), rollout via can_eval.rollout_eval_can (@ STEPS pas, max 300).

  python -u experiments/can/02_eval_baseline.py            # tous les checkpoints, 50 val
  python -u experiments/can/02_eval_baseline.py --smoke     # dernier checkpoint, 5 départs
Sortie : results/runs/can/01_baseline_eval.json
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import lift_eval, can_eval
from src.mini_cnn import load_minicnn_from_ckpt

RUN_DIR = Path("results/runs/can/01_baseline")
DATASET_REPO, DATASET_ROOT = "local/can_ph", "data_cache/lerobot_can_ph"
STEPS = 10
OUT = Path("results/runs/can/01_baseline_eval.json")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = can_eval.load_or_make_split()["val"]              # démos 150-199 (held-out)
    init_states = can_eval.load_init_states(val_idx)
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def procs(ckpt):
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return pre, post

    ckpt_root = RUN_DIR / "checkpoints"
    steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    if args.smoke:
        steps_ck = steps_ck[-1:]
        init_states = init_states[:5]
    print(f"Checkpoints: {steps_ck} | val states: {len(init_states)}", flush=True)

    # val-loss propre par checkpoint
    val_ds = None
    val_by_step = {}
    for s in steps_ck:
        ckpt = str(ckpt_root / f"{s:06d}" / "pretrained_model")
        policy = load_minicnn_from_ckpt(ckpt, device)
        pre, _ = procs(ckpt)
        if val_ds is None:
            val_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                    delta_timestamps=build_delta_timestamps(policy.config, fps))
        dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                bs = batch["action"].shape[0]
                loss, _ = policy.forward(pre(batch)); tot += float(loss) * bs; n += bs
        val_by_step[s] = tot / n
        print(f"  step {s}: val_loss={val_by_step[s]:.4f}", flush=True)
        del policy
    best = min(val_by_step, key=val_by_step.get)
    print(f"-> meilleur checkpoint {best} (val_loss={val_by_step[best]:.4f})", flush=True)

    # rollout @ STEPS pas du best
    ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
    policy = load_minicnn_from_ckpt(ckpt, device)
    pre, post = procs(ckpt)
    per, agg = can_eval.rollout_eval_can(policy, pre, post, init_states, device=device,
                                         num_inference_steps=STEPS, stop_on_success=True)
    k = sum(e["success"] for e in per)
    lo, hi = wilson_ci(k, len(per))
    print(f"\n=== BASELINE CAN : {k}/{len(per)} = {agg['success_rate']:.1%} [{lo:.1%}, {hi:.1%}] "
          f"t_succ_med={agg['t_success_median']} (best step {best}, @{STEPS} pas) ===", flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps({"best_step": best, "val_loss": val_by_step[best], "steps": STEPS,
                                   "n": len(per), "n_success": k, "success_rate": agg["success_rate"],
                                   "ci95": [lo, hi], "t_success_median": agg["t_success_median"],
                                   "val_by_step": val_by_step}, indent=2))
        print(f"JSON: {OUT}")


if __name__ == "__main__":
    main()
