"""Phase 4 / Phase 0 — val noise-MSE d'un checkpoint Diffusion sur le VAL set figé.

Détecteur d'overfit : la loss de prédiction de bruit (diffusion) sur les 50 démos
jamais entraînées. Quand elle remonte d'un checkpoint au suivant → début d'overfit
→ signal d'arrêt. ⚠️ Proxy BRUITÉ du succès rollout : à croiser avec 47_phase0_eval.py.

Usage :
  python -u experiments/lift/48_val_loss.py \
      --checkpoint results/runs/lift/47_baseline_150/checkpoints/006000/pretrained_model
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import lift_eval

DATASET_REPO = "local/lift_ph"
DATASET_ROOT = "data_cache/lerobot_lift_ph"


def build_delta_timestamps(cfg, fps):
    """Reconstruit les delta_timestamps attendus par le dataset depuis la config policy."""
    dt = {}
    for key in list(cfg.input_features):  # observation.image, observation.state
        dt[key] = [i / fps for i in cfg.observation_delta_indices]
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-batches", type=int, default=None, help="pour un smoke test rapide")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    ckpt = args.checkpoint
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )

    val_idx = lift_eval.load_or_make_split()["val"]
    # fps depuis le meta du dataset
    meta = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))
    fps = meta["fps"]
    delta_ts = build_delta_timestamps(policy.config, fps)
    print(f"Val episodes: {len(val_idx)} | fps={fps} | delta_timestamps keys={list(delta_ts)}", flush=True)

    ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx, delta_timestamps=delta_ts)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Val frames: {len(ds)} | batches: {len(dl)}", flush=True)

    total, n = 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if args.limit_batches and bi >= args.limit_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = pre(batch)
            loss, _ = policy.forward(batch)
            total += float(loss) * args.batch_size
            n += args.batch_size
            if bi % 5 == 0:
                print(f"  batch {bi+1}/{len(dl)} : running val_loss={total/n:.4f}", flush=True)

    val_loss = total / n
    print(f"\n=== val noise-MSE = {val_loss:.4f}  ({n} samples) ===")
    out = Path(args.out) if args.out else Path(ckpt).parent.parent / "val_loss.json"
    out.write_text(json.dumps({"checkpoint": ckpt, "val_loss": val_loss, "n_samples": n}, indent=2))
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
