"""val_loss FULL (1 seed, tout le set val) pour checkpoints mini-CNN — lisse, vs la val_loss
live (1 batch) trop bruitee. Parcourt plusieurs run-dirs, tous les --interval steps.

Usage :
  venv312/bin/python -u experiments/phase5_methodology/26_val_loss_full_minicnn.py \
    --run-dirs mini_cosine,mini_cosine_continue,mini_cosine_continue2,mini_cosine_150k \
    --device mps --out results/runs/phase5_methodology/val_cosine_full.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
import torch
from torch.utils.data import DataLoader
from src import can_eval
from src.mini_cnn import load_minicnn_from_ckpt

PHASE = Path("results/runs/phase5_methodology")
DATASET_REPO = "local/can_ph_proprio_birdview"
DATASET_ROOT = "data_cache/lerobot_can_ph_proprio_birdview"
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


def steps_done(path):
    d = set()
    if Path(path).exists():
        for r in csv.DictReader(open(path)): d.add(int(r["step"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True, help="dossiers de run separes par virgules (relatifs a phase5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--interval", type=int, default=1000)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    val_idx = can_eval.load_or_make_split()["val"]
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]
    done = steps_done(a.out)

    # collecte des checkpoints (step -> path) sur tous les run-dirs, tous les `interval`
    targets = {}
    for rd in a.run_dirs.split(","):
        ckroot = PHASE / rd / "checkpoints"
        if not ckroot.exists(): continue
        for p in sorted(ckroot.iterdir()):
            if p.is_dir() and p.name.isdigit():
                step = int(p.name)
                if step % a.interval == 0 and step not in done and (p / "pretrained_model" / "model.safetensors").exists():
                    targets[step] = p / "pretrained_model"
    targets = sorted(targets.items())
    print(f"[26] device={device}, val={len(val_idx)} ep, {len(targets)} checkpoints a mesurer", flush=True)

    val_ds = None
    for step, ckdir in targets:
        ckpt = str(ckdir)
        try:
            policy = load_minicnn_from_ckpt(ckpt, device); policy.eval()
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            if val_ds is None:
                val_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                        delta_timestamps=build_delta_timestamps(policy.config, fps))
            t0 = time.time(); vl = compute_val_loss(policy, pre, val_ds, device, SEED); el = time.time() - t0
            wh = not Path(a.out).exists()
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", "val_loss_full", "elapsed_sec"])
                if wh: w.writeheader()
                w.writerow({"step": step, "val_loss_full": vl, "elapsed_sec": el})
            print(f"[26] step={step}: val_loss_full={vl:.4f} ({el:.0f}s)", flush=True)
            del policy
        except Exception as e:
            print(f"[26] erreur step={step}: {e}", flush=True)
    print("[26] DONE", flush=True)


if __name__ == "__main__":
    main()
