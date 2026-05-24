"""Phase 5 — efficacité données : combien de démos pour tenir 100 % ?

Même modèle final (mini-CNN + U-Net [32,64,128]) entraîné sur N = 150/100/50/20/10 démos
(préfixes 0..N-1 ; val = 150-199 figé, held-out pour tous). Pour chaque N : val-loss propre
→ meilleur checkpoint → rollout @ 4 pas sur les 50 val. Courbe succès vs nb démos.

Sortie : results/runs/lift/63_data_efficiency.{json,png}
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

DATASET_REPO, DATASET_ROOT = "local/lift_ph", "data_cache/lerobot_lift_ph"
STEPS = 4  # pas de diffusion (point de fonctionnement)
OUT = Path("results/runs/lift/63_data_efficiency")
# (N démos, run dir)
RUNS = [
    (150, "results/runs/lift/61_minicnn"),
    (100, "results/runs/lift/63_dataeff_n100"),
    (50, "results/runs/lift/63_dataeff_n50"),
    (20, "results/runs/lift/63_dataeff_n20"),
    (10, "results/runs/lift/63_dataeff_n10"),
]


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    val_ds = None
    results = []
    for N, run_dir in RUNS:
        ckpt_root = Path(run_dir) / "checkpoints"
        if not ckpt_root.exists():
            print(f"  N={N}: {ckpt_root} absent, skip", flush=True); continue
        steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
        print(f"\n##### N={N} démos ({run_dir}) — checkpoints {steps_ck} #####", flush=True)

        # val-loss propre par checkpoint -> meilleur
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
            del policy
        best = min(val_by_step, key=val_by_step.get)
        print(f"  -> meilleur checkpoint {best} (val_loss={val_by_step[best]:.4f})", flush=True)

        # rollout @ STEPS pas du meilleur
        ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
        policy = load_minicnn_policy(ckpt, device)
        pre, post = procs(ckpt)
        _, agg = lift_eval.rollout_eval(policy, pre, post, env, init_states,
            device=device, num_inference_steps=STEPS, verbose=False)
        del policy
        rec = {"n_demos": N, "best_step": best, "val_loss": val_by_step[best], "steps": STEPS, **agg}
        results.append(rec)
        print(f"  => N={N} : succès {agg['success_rate']:.0%} t_succ={agg['t_success_median']} "
              f"max_z={agg['max_z_mean']:.3f} (val {val_by_step[best]:.4f})", flush=True)
        (OUT.with_suffix(".json")).write_text(json.dumps({"steps": STEPS, "results": results}, indent=2))

    # plot succès vs N
    results.sort(key=lambda r: r["n_demos"])
    xs = [r["n_demos"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [r["success_rate"] * 100 for r in results], "o-", color="#d62728")
    for r in results:
        ax.annotate(f"{r['success_rate']:.0%}", (r["n_demos"], r["success_rate"] * 100),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("nb de démos d'entraînement"); ax.set_ylabel("succès (%) @ 4 pas, 50 val ép.")
    ax.set_ylim(-5, 105); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    ax.set_title("Efficacité données — mini-CNN [32,64,128] sur Lift")
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"\nJSON: {OUT.with_suffix('.json')}\nPlot: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
