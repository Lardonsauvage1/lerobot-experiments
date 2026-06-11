"""Phase 5 — val_loss LÉGÈRE : 1 seed, tous les 1000 steps.

Version allégée du 05 (qui faisait 6 passes/ckpt tous les 100 steps).
On a constaté que val_loss 1seed ≈ 5seeds -> on garde 1 seed seulement.
Passe unique (l'entraînement dense est terminé, tous les ckpts existent).
Saute les steps déjà mesurés (dans val_losses.csv OU dans la sortie ci-dessous).

Sortie : results/runs/phase5_methodology/26_resnet34_dense/val_loss_1seed_fast.csv
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import can_eval

RUN_DIR = Path("results/runs/phase5_methodology/26_resnet34_dense")
CKPT_ROOT = RUN_DIR / "checkpoints"
OUT_CSV = RUN_DIR / "val_loss_1seed_fast.csv"
PREV_CSV = RUN_DIR / "val_losses.csv"      # données du 05 (tous les 100) à ne pas refaire
DATASET_REPO = "local/can_ph_proprio_birdview"
DATASET_ROOT = "data_cache/lerobot_can_ph_proprio_birdview"
EVAL_INTERVAL = 1000     # tous les 1000 steps
SEED = 0


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


def steps_in_csv(path, col="step"):
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add(int(row[col]))
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
    print(f"[08] device={device}, val_idx={len(val_idx)} ép.", flush=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    done = steps_in_csv(PREV_CSV) | steps_in_csv(OUT_CSV)
    print(f"[08] {len(done)} steps déjà mesurés (05 + fast), on saute ceux-là", flush=True)

    targets = []
    for p in sorted(CKPT_ROOT.iterdir()):
        if p.is_dir() and p.name.isdigit():
            step = int(p.name)
            if step % EVAL_INTERVAL == 0 and step not in done and (p / "pretrained_model").exists():
                targets.append((step, p))
    print(f"[08] {len(targets)} checkpoints à mesurer : {[s for s,_ in targets]}", flush=True)

    val_ds = None
    for step, p in targets:
        ckpt = str(p / "pretrained_model")
        try:
            policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            if val_ds is None:
                val_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                        delta_timestamps=build_delta_timestamps(policy.config, fps))
            t0 = time.time()
            vl = compute_val_loss(policy, pre, val_ds, device, seed=SEED)
            elapsed = time.time() - t0
            append_csv({"step": step, "val_loss_1seed": vl, "elapsed_sec": elapsed})
            print(f"[08] step={step}: val_loss={vl:.4f} ({elapsed:.0f}s)", flush=True)
            del policy
        except Exception as e:
            print(f"[08] ⚠ erreur step={step} : {e}", flush=True)
    print("[08] VAL_LOSS 1SEED DONE", flush=True)


if __name__ == "__main__":
    main()
