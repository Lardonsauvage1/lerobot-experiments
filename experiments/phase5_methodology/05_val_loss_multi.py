"""Phase 5 — Calcule val_loss avec multiples méthodes sur chaque ckpt sauvé.

Pour chaque ckpt:
- val_loss_full_1seed : moyenne sur les 50 ép. val (batch 32, 1 seed dataloader)
- val_loss_full_5seeds : 5 mesures avec seeds différents -> mean ± std

Sortie : results/runs/phase5_methodology/26_resnet34_dense/val_losses.csv
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import can_eval

RUN_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense")
CKPT_ROOT = RUN_DIR / "checkpoints"
OUT_CSV = RUN_DIR / "val_losses.csv"
DATASET_REPO = "local/can_ph_proprio_birdview"
DATASET_ROOT = "data_cache/lerobot_can_ph_proprio_birdview"
EVAL_INTERVAL = 100      # tous les ckpts (100 step)
N_SEEDS = 5              # nb de seeds dataloader pour mesure 5seeds


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def compute_val_loss(policy, pre, val_ds, device, seed):
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0, generator=g, drop_last=True)
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bs = batch["action"].shape[0]
            loss, _ = policy.forward(pre(batch)); tot += float(loss) * bs; n += bs
    return tot / n


def load_done():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(int(row["step"]))
    return done


def append_csv(row):
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = can_eval.load_or_make_split()["val"]
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]
    print(f"[05] device={device}, val_idx={len(val_idx)} ép.", flush=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    done = load_done()
    print(f"[05] {len(done)} ckpts déjà mesurés", flush=True)

    val_ds = None
    while True:
        avail = []
        if CKPT_ROOT.exists():
            for p in sorted(CKPT_ROOT.iterdir()):
                if not p.is_dir() or not p.name.isdigit():
                    continue
                step = int(p.name)
                if step % EVAL_INTERVAL == 0 and step not in done:
                    if (p / "pretrained_model").exists() and (p / "training_state").exists():
                        avail.append((step, p))

        if not avail:
            if Path("/tmp/stop_05").exists():
                print(f"[05] stop sentinel"); break
            time.sleep(60); continue

        step, p = avail[0]
        ckpt = str(p / "pretrained_model")
        try:
            policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            if val_ds is None:
                val_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                        delta_timestamps=build_delta_timestamps(policy.config, fps))
            losses = []
            t0 = time.time()
            for seed in range(N_SEEDS):
                losses.append(compute_val_loss(policy, pre, val_ds, device, seed=seed))
            elapsed = time.time() - t0
            mean = float(np.mean(losses))
            std = float(np.std(losses))
            row = {"step": step, "val_loss_full_1seed": losses[0],
                   "val_loss_full_5seeds_mean": mean, "val_loss_full_5seeds_std": std,
                   "elapsed_sec": elapsed}
            append_csv(row)
            done.add(step)
            if step % 1000 == 0:
                print(f"[05] step={step}: mean={mean:.4f} std={std:.4f} ({elapsed:.1f}s)", flush=True)
            del policy
        except Exception as e:
            print(f"[05] ⚠ erreur step={step} : {e}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
