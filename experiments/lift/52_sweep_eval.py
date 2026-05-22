"""Phase 4 — éval comparée du sweep U-Net : trouve le plus petit modèle qui tient le baseline.

Pour chaque run-dir (taille de U-Net) :
  1. val noise-MSE (set val complet) sur CHAQUE checkpoint → choisit le min (point d'arrêt propre)
  2. rollout du meilleur checkpoint (50 ép. val figées) → succès / t_success / max_z / hold
  3. compte les params du modèle
Sauve un JSON récap + trace le Pareto params vs (succès, t_success, max_z).

Usage :
  python -u experiments/lift/52_sweep_eval.py \
    --run-dirs results/runs/lift/51_unet_d256_512_1024 \
               results/runs/lift/51_unet_d128_256_512 \
               results/runs/lift/51_unet_d64_128_256
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src import lift_eval

DATASET_REPO = "local/lift_ph"
DATASET_ROOT = "data_cache/lerobot_lift_ph"
OUT = Path("results/runs/lift/51_unet_sweep_eval")

# Référence baseline 263.7M (run 47, checkpoint 6000) pour le Pareto
BASELINE = {"label": "baseline 252M U-Net", "params": 263.7e6,
            "success_rate": 1.0, "t_success_median": 43.0, "max_z_mean": 1.033}


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def full_val_loss(policy, pre, val_ds, device):
    dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bs = batch["action"].shape[0]
            loss, _ = policy.forward(pre(batch))
            total += float(loss) * bs; n += bs
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--inference-steps", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]
    OUT.mkdir(parents=True, exist_ok=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def load(ckpt):
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return policy, pre, post

    # incrémental : on repart des résultats déjà calculés (clé = nom du run)
    ej = OUT / "sweep_eval.json"
    by_run = {}
    if ej.exists():
        for r in json.loads(ej.read_text()).get("results", []):
            by_run[r["run"]] = r

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        ckpt_root = run_dir / "checkpoints"
        steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
        print(f"\n##### {run_dir.name} — checkpoints {steps} #####", flush=True)

        # 1) val-loss propre par checkpoint -> meilleur
        val_dl_ds = None
        val_by_step = {}
        for step in steps:
            ckpt = str(ckpt_root / f"{step:06d}" / "pretrained_model")
            policy, pre, post = load(ckpt)
            if val_dl_ds is None:
                dt = build_delta_timestamps(policy.config, fps)
                val_dl_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx, delta_timestamps=dt)
            vl = full_val_loss(policy, pre, val_dl_ds, device)
            val_by_step[step] = vl
            print(f"  step {step}: val_loss={vl:.4f}", flush=True)
            del policy, pre, post
        best_step = min(val_by_step, key=val_by_step.get)
        print(f"  -> meilleur checkpoint (val min) : {best_step} (val_loss={val_by_step[best_step]:.4f})", flush=True)

        # 2) rollout du meilleur
        ckpt = str(ckpt_root / f"{best_step:06d}" / "pretrained_model")
        policy, pre, post = load(ckpt)
        params = sum(p.numel() for p in policy.parameters())
        _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
            device=device, num_inference_steps=args.inference_steps)
        down_dims = list(policy.config.down_dims)
        del policy, pre, post

        rec = {"run": run_dir.name, "down_dims": down_dims, "params": params,
               "best_step": best_step, "val_loss": val_by_step[best_step], **agg}
        by_run[run_dir.name] = rec
        print(f"  => {run_dir.name}: {params/1e6:.1f}M params | success={agg['success_rate']:.0%} "
              f"t_succ={agg['t_success_median']} max_z={agg['max_z_mean']:.3f} hold={agg['hold_fraction_mean']:.2f}", flush=True)
        results = sorted(by_run.values(), key=lambda r: -r["params"])
        ej.write_text(json.dumps({"baseline": BASELINE, "results": results}, indent=2))

    # Pareto
    results = sorted(by_run.values(), key=lambda r: -r["params"])
    pts = [BASELINE] + results
    xs = [p["params"] / 1e6 for p in pts]
    labels = [p.get("label") or f"{p['down_dims']}" for p in pts]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in [(axes[0], "success_rate", "Success rate"),
                           (axes[1], "t_success_median", "t_success médian (↓ mieux)"),
                           (axes[2], "max_z_mean", "max_z (marge, ↑ mieux)")]:
        # filtre les None (ex. t_success quand 0% succès) pour ne pas casser le tracé
        trip = [(p["params"] / 1e6, (p[key] * 100 if key == "success_rate" else p[key]), lab)
                for p, lab in zip(pts, labels) if p.get(key) is not None]
        if trip:
            xs_, ys_, labs_ = zip(*trip)
            ax.plot(xs_, ys_, "o", markersize=10)
            for x, y, lab in zip(xs_, ys_, labs_):
                ax.annotate(lab, (x, y), fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")
        ax.set_xscale("log"); ax.set_xlabel("Params (M, log)"); ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 105)
    fig.suptitle("Sweep U-Net — Pareto taille vs perfs (50 val ep)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_pareto.png", dpi=110)
    print(f"\nJSON: {OUT/'sweep_eval.json'}\nPlot: {OUT/'sweep_pareto.png'}")


if __name__ == "__main__":
    main()
