"""Lift vision-only (image + proprio 9D, SANS coords cube) — courbe efficacité-données.

Combien de démos suffisent quand le modèle doit VOIR le cube (pas de coords) ? Pendant de la
courbe phase 4 (qui DONNAIT les coords, ~20 démos ≈ 98%). Même arche pour tous : ResNet18 +
U-Net [64,128,256], état 9D. Éval RIGOUREUSE 500 rollouts (réutilise phase4_eval500.npy) + Wilson.

N=150 = results/runs/lift/73_proprio ; N<150 = results/runs/lift/73_proprio_n{N}.
Sortie : results/runs/lift/proprio_dataeff_500.{json,png}
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import lift_eval

DATASET_REPO, DATASET_ROOT = "local/lift_ph_proprio", "data_cache/lerobot_lift_ph_proprio"
STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
STEPS = 10
NS = [150, 100, 50, 20, 10]
OUT = Path("results/runs/lift/proprio_dataeff_500")


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


def run_dir_for(N):
    return Path("results/runs/lift/73_proprio") if N == 150 else Path(f"results/runs/lift/73_proprio_n{N}")


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = lift_eval.load_or_make_split()["val"]
    states = np.load(STATES_PATH)
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]

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

    results = []
    if OUT.with_suffix(".json").exists():
        results = json.loads(OUT.with_suffix(".json").read_text()).get("results", [])
    done = {r["n_demos"] for r in results}

    val_ds = None
    for N in NS:
        if N in done:
            print(f"N={N} déjà fait, skip", flush=True); continue
        ckpt_root = run_dir_for(N) / "checkpoints"
        if not ckpt_root.exists():
            print(f"N={N}: {ckpt_root} absent, skip", flush=True); continue
        steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
        print(f"\n##### N={N} ({ckpt_root.parent.name}) — checkpoints {steps_ck} #####", flush=True)
        # best par val-loss (val = 50 ép. figées 150-199)
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
            del policy
        best = min(val_by_step, key=val_by_step.get)
        print(f"  best {best} (val_loss={val_by_step[best]:.4f})", flush=True)
        # rollout 500 @ STEPS pas
        ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
        policy, pre, post = load(ckpt)
        per, agg = lift_eval.rollout_eval_chunked(policy, pre, post, states, device=device,
            num_inference_steps=STEPS, stop_on_success=True,
            fix_obs_fn=lambda o: o, state_fn=build_state_vector_proprio)
        del policy
        k = sum(e["success"] for e in per)
        lo, hi = wilson_ci(k, len(per))
        rec = {"n_demos": N, "best_step": best, "val_loss": val_by_step[best], "steps": STEPS,
               "n": len(per), "n_success": k, "success_rate": agg["success_rate"],
               "ci95_low": lo, "ci95_high": hi, "t_success_median": agg["t_success_median"]}
        results.append(rec)
        print(f"  => N={N} : {k}/{len(per)} = {agg['success_rate']:.1%} [{lo:.1%}, {hi:.1%}]", flush=True)
        OUT.with_suffix(".json").write_text(json.dumps({"steps": STEPS, "n_eval": int(states.shape[0]),
            "mode": "vision-only (9D proprio, sans coords cube)", "results": results}, indent=2))

    # plot succès vs N
    results.sort(key=lambda r: r["n_demos"])
    xs = [r["n_demos"] for r in results]
    ys = [r["success_rate"] * 100 for r in results]
    lo = [(r["success_rate"] - r["ci95_low"]) * 100 for r in results]
    hi = [(r["ci95_high"] - r["success_rate"]) * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="#8c564b", capsize=4, label="vision-only (9D)")
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("nb de démos"); ax.set_ylabel("succès (%) @ 10 pas, 500 rollouts (IC95 Wilson)")
    ax.set_ylim(0, 102); ax.grid(True, alpha=0.3)
    ax.set_title("Lift efficacité-données EN VISION PURE (sans coords du cube)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=120)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
