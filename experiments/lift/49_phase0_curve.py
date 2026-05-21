"""Phase 4 / Phase 0 — courbe métriques vs steps sur tous les checkpoints d'un run.

Pour chaque checkpoint : rollout (succès, t_success, max_z, hold) sur le val set figé
+ val noise-MSE. Sauve un JSON récap et trace une figure multi-panneaux → sert à
choisir le point d'arrêt (quand les métriques rollout plafonnent / la val-loss remonte).

Usage :
  python -u experiments/lift/49_phase0_curve.py --run-dir results/runs/lift/47_baseline_150 --episodes 50
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


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=50, help="nb d'épisodes val (défaut: tous = 50)")
    ap.add_argument("--inference-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    run_dir = Path(args.run_dir)
    ckpt_root = run_dir / "checkpoints"
    steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    print(f"Checkpoints: {steps}", flush=True)

    split = lift_eval.load_or_make_split()
    val_idx = split["val"][:args.episodes]
    init_states = lift_eval.load_init_states(val_idx)
    env = lift_eval.make_env()

    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def compute_val_loss(policy, pre):
        dt = build_delta_timestamps(policy.config, fps)
        ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx, delta_timestamps=dt)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                bs = batch["action"].shape[0]
                loss, _ = policy.forward(pre(batch))
                total += float(loss) * bs; n += bs
        return total / n

    out_json = run_dir / "phase0_curve.json"
    out_png = run_dir / "phase0_curve.png"

    def save():
        out_json.write_text(json.dumps({"episodes": len(val_idx), "inference_steps": args.inference_steps,
                                        "val_episodes": val_idx, "results": results}, indent=2))

    results = []
    for step in steps:
        ckpt = str(ckpt_root / f"{step:06d}" / "pretrained_model")
        print(f"\n===== checkpoint {step} =====", flush=True)
        try:
            policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
                to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
            _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
                device=device, max_steps=args.max_steps, num_inference_steps=args.inference_steps)
            try:
                val_loss = compute_val_loss(policy, pre)
            except Exception as e:
                print(f"  [val_loss KO] {e}", flush=True); val_loss = None
            results.append({"step": step, "val_loss": val_loss, **agg})
            vl = "None" if val_loss is None else f"{val_loss:.4f}"
            print(f"  -> success={agg['success_rate']:.0%} t_succ={agg['t_success_median']} "
                  f"max_z={agg['max_z_mean']:.3f} hold={agg['hold_fraction_mean']:.2f} val_loss={vl}", flush=True)
            del policy, pre, post
        except Exception as e:
            print(f"  [checkpoint {step} KO] {e}", flush=True)
        save()  # sauvegarde incrémentale après chaque checkpoint
    print(f"\nJSON: {out_json}")

    # plot (robuste aux None : on filtre par série)
    def series(key, scale=1.0):
        pts = [(r["step"], r[key] * scale) for r in results if r.get(key) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(*series("success_rate", 100), "o-", color="#d62728")
    axes[0, 0].set_title("Success rate (%)"); axes[0, 0].set_ylim(-5, 105)
    axes[0, 1].plot(*series("t_success_median"), "o-", color="#1f77b4")
    axes[0, 1].set_title("t_success médian (steps, ↓ = plus décisif)")
    axes[1, 0].plot(*series("max_z_mean"), "o-", color="#2ca02c", label="max_z")
    axes[1, 0].plot(*series("hold_fraction_mean"), "s--", color="#ff7f0e", label="hold")
    axes[1, 0].set_title("Marge de levage / maintien"); axes[1, 0].legend()
    axes[1, 1].plot(*series("val_loss"), "o-", color="#9467bd")
    axes[1, 1].set_title("Val noise-MSE (↑ = overfit)")
    for ax in axes.flat:
        ax.set_xlabel("Training step"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Phase 0 — {run_dir.name} : métriques vs steps ({len(val_idx)} val ep)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"Plot: {out_png}")


if __name__ == "__main__":
    main()
