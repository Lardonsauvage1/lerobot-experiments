"""Éval 1 cellule de la grille vision-pure (mini-CNN, état 9D, 3×5 grid).

Pour 1 run-dir :
  - val-loss/checkpoint -> meilleur
  - rollout 500 (init states figés `phase4_eval500.npy`, mêmes que phase 4)
  - écrit <run-dir>_eval500.json avec succès, IC95 Wilson, t_med, params, val_loss

Usage : python 81_visionpure_cell_eval.py --run-dir results/runs/lift_visionpure/grid_d32_64_128_n150
Sortie : <run-dir>_eval500.json
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

from src import lift_eval
from src.mini_cnn import load_minicnn_from_ckpt

DATASET_REPO, DATASET_ROOT = "local/lift_ph_proprio", "data_cache/lerobot_lift_ph_proprio"
STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
STEPS = 10


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_state_vector_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="default = <run-dir>_eval500.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / f"{run_dir.name}_eval500.json"

    device = torch.device("cpu")  # atomman = CPU
    val_idx = lift_eval.load_or_make_split()["val"]
    states = np.load(STATES_PATH)
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

    ckpt_root = run_dir / "checkpoints"
    steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    print(f"[{run_dir.name}] checkpoints : {steps_ck}", flush=True)

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
    print(f"  -> best {best} (val_loss={val_by_step[best]:.4f})", flush=True)

    ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
    policy = load_minicnn_from_ckpt(ckpt, device)
    pre, post = procs(ckpt)
    params = sum(p.numel() for p in policy.parameters())
    per, agg = lift_eval.rollout_eval_chunked(policy, pre, post, states, device=device,
        num_inference_steps=STEPS, stop_on_success=True,
        fix_obs_fn=lambda o: o, state_fn=build_state_vector_proprio)
    k = sum(e["success"] for e in per)
    lo, hi = wilson_ci(k, len(per))
    rec = {"run_dir": str(run_dir), "params": params, "best_step": best,
           "val_loss": val_by_step[best], "steps": STEPS,
           "n": len(per), "n_success": k, "success_rate": agg["success_rate"],
           "ci95_low": lo, "ci95_high": hi,
           "t_success_median": agg["t_success_median"],
           "val_by_step": val_by_step}
    out.write_text(json.dumps(rec, indent=2))
    print(f"\n=> {run_dir.name} : {k}/{len(per)} = {agg['success_rate']:.1%} [{lo:.1%}, {hi:.1%}]", flush=True)
    print(f"JSON: {out}", flush=True)


if __name__ == "__main__":
    main()
