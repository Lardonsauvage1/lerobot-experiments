"""Lift "image + proprio SEULE" (sans coords du cube) — éval de transférabilité réel.

Compare au baseline Lift habituel (qui donnait la pose complète du cube, ~99 %) : ici l'état
est 9D = proprio seule (eef_pos+eef_quat+gripper), le modèle doit VOIR le cube dans l'image.
Vision ResNet18, U-Net [64,128,256]. val-loss/checkpoint → best → rollout 50 val @ 10 pas.

Sortie : results/runs/lift/73_proprio_eval.json
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

DATASET_REPO, DATASET_ROOT = "local/lift_ph_proprio", "data_cache/lerobot_lift_ph_proprio"
STEPS = 10


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_state_vector_proprio(obs):
    """9D : proprio seule, SANS object (ni coords du cube)."""
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
    ap.add_argument("--run-dir", default="results/runs/lift/73_proprio")
    ap.add_argument("--out", default="results/runs/lift/73_proprio_eval.json")
    args = ap.parse_args()
    RUN_DIR = Path(args.run_dir)
    OUT = Path(args.out)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = lift_eval.load_or_make_split()["val"]
    init_states = lift_eval.load_init_states(val_idx)
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def load(ckpt):
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()  # ResNet18, pas de swap
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return policy, pre, post

    ckpt_root = RUN_DIR / "checkpoints"
    steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    print(f"Checkpoints: {steps_ck}", flush=True)

    val_ds = None
    val_by_step = {}
    for s in steps_ck:
        ckpt = str(ckpt_root / f"{s:06d}" / "pretrained_model")
        policy, pre, _ = load(ckpt)
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

    ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
    policy, pre, post = load(ckpt)
    per, agg = lift_eval.rollout_eval_chunked(
        policy, pre, post, init_states, device=device, num_inference_steps=STEPS,
        stop_on_success=True, fix_obs_fn=lambda o: o, state_fn=build_state_vector_proprio)
    k = sum(e["success"] for e in per)
    lo, hi = wilson_ci(k, len(per))
    print(f"\n=== LIFT image+proprio (9D, SANS cube) : {k}/{len(per)} = {agg['success_rate']:.1%} "
          f"[{lo:.1%}, {hi:.1%}]  t_succ_med={agg['t_success_median']} (best {best}, @{STEPS} pas) ===", flush=True)
    OUT.write_text(json.dumps({"best_step": best, "val_loss": val_by_step[best], "steps": STEPS,
                               "n": len(per), "n_success": k, "success_rate": agg["success_rate"],
                               "ci95": [lo, hi], "t_success_median": agg["t_success_median"],
                               "val_by_step": val_by_step}, indent=2))
    print(f"JSON: {OUT}")


if __name__ == "__main__":
    main()
