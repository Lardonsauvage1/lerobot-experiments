"""Phase 4 — éval du modèle mini-CNN (vision ~0.03M au lieu de ResNet18 11.2M).

Charge via load_minicnn_policy (la config dit resnet18 → swap obligatoire avant load).
Pour chaque checkpoint : val-loss propre → meilleur ; puis rollout (50 val ép.) + latence
aux pas qui comptent (10 et 4). Compare au [32,64,128] + ResNet (12.8M, 49ms@4pas, 100%).

Sortie : results/runs/lift/61_minicnn/minicnn_eval.json
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

RUN_DIR = Path("results/runs/lift/61_minicnn")
DATASET_REPO, DATASET_ROOT = "local/lift_ph", "data_cache/lerobot_lift_ph"
STEPS_EVAL = [10, 4]


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def fake_obs(device):
    return {"observation.image": torch.rand(1, 3, 96, 96, device=device),
            "observation.state": torch.rand(1, 19, device=device)}


def measure_latency(policy, device, steps, n=6):
    policy.diffusion.num_inference_steps = steps
    sync = (lambda: torch.mps.synchronize()) if device.type == "mps" else (lambda: None)
    with torch.no_grad():
        for _ in range(2):
            policy.reset(); policy.select_action(fake_obs(device)); sync()
        ts = []
        for _ in range(n):
            policy.reset(); sync(); t0 = time.perf_counter()
            policy.select_action(fake_obs(device)); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt_root = RUN_DIR / "checkpoints"
    steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    print(f"Checkpoints: {steps_ck}", flush=True)

    val_idx = lift_eval.load_or_make_split()["val"]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()
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

    # 1) val-loss propre par checkpoint -> meilleur
    val_ds = None
    val_by_step = {}
    for s in steps_ck:
        ckpt = str(ckpt_root / f"{s:06d}" / "pretrained_model")
        policy = load_minicnn_policy(ckpt, device)
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
    print(f"  -> meilleur checkpoint : {best} (val_loss={val_by_step[best]:.4f})", flush=True)

    # 2) rollout + latence du meilleur, aux pas qui comptent
    ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
    policy = load_minicnn_policy(ckpt, device)
    pre, post = procs(ckpt)
    params = sum(p.numel() for p in policy.parameters())
    results = []
    for st in STEPS_EVAL:
        lat = measure_latency(policy, device, st)
        _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
            device=device, num_inference_steps=st, verbose=False)
        results.append({"steps": st, "latency_ms": lat, **agg})
        print(f"  pas={st} : {lat:.1f}ms · succès {agg['success_rate']:.0%} "
              f"t_succ={agg['t_success_median']} max_z={agg['max_z_mean']:.3f}", flush=True)

    out = {"params": params, "best_step": best, "val_loss": val_by_step[best],
           "val_by_step": val_by_step, "results": results,
           "ref_resnet_32_64_128": {"params": 12.8e6, "steps4": {"latency_ms": 49, "success": 1.0}}}
    (RUN_DIR / "minicnn_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== mini-CNN : {params/1e6:.2f}M params (vs 12.8M resnet) ===")
    print(f"JSON: {RUN_DIR/'minicnn_eval.json'}")


if __name__ == "__main__":
    main()
