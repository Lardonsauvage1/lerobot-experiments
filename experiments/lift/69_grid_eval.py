"""Phase 4 — grille (taille U-Net × données), éval RIGOUREUSE à 500 rollouts.

Évalue les 10 NOUVEAUX modèles mini-CNN entraînés par run_69_grid.sh :
  U-Net ∈ {[64,128,256], [128,256,512]} × N ∈ {150,100,50,20,10}.
(Les 5 cellules [32,64,128] × N sont fournies par 67_dataeff_500.py → grille assemblée à la main.)

Pour chaque modèle, comme 63 : val-loss propre par checkpoint (val = 50 ép. figées 150-199)
→ meilleur checkpoint → rollout @ 4 pas sur les MÊMES 500 départs que 67 (phase4_eval500.npy),
arrêt au 1er succès, IC95 de Wilson.

Sortie : results/runs/lift/69_grid_500.json
"""

import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import lift_eval
from src.mini_cnn import load_minicnn_policy

DATASET_REPO, DATASET_ROOT = "local/lift_ph", "data_cache/lerobot_lift_ph"
STEPS = 4
STATES_PATH = Path("results/runs/lift/phase4_eval500.npy")
OUT = Path("results/runs/lift/69_grid_500.json")
DIMS = [("[64,128,256]", "64_128_256"), ("[128,256,512]", "128_256_512")]
NS = [150, 100, 50, 20, 10]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if not STATES_PATH.exists():
        raise SystemExit(f"{STATES_PATH} absent — 67_dataeff_500.py doit l'avoir généré.")
    states = np.load(STATES_PATH)
    print(f"États de test (partagés avec 67) : {states.shape}", flush=True)
    val_idx = lift_eval.load_or_make_split()["val"]  # 150-199, figé pour la val-loss
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]
    # env de rollout recréé par tranche dans rollout_eval_chunked (anti-dégradation renderer)

    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def procs(ckpt):
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return pre, post

    # résultats déjà calculés (incrémental, clé = (down_dims, n_demos))
    results = []
    if OUT.exists():
        results = json.loads(OUT.read_text()).get("results", [])
    done = {(r["down_dims"], r["n_demos"]) for r in results}

    val_ds = None
    for dims, dtag in DIMS:
        for N in NS:
            if (dims, N) in done:
                print(f"  {dims} N={N} déjà fait, skip", flush=True); continue
            run_dir = Path(f"results/runs/lift/69_grid_d{dtag}_n{N}")
            ckpt_root = run_dir / "checkpoints"
            if not ckpt_root.exists():
                print(f"  {dims} N={N}: {ckpt_root} absent, skip", flush=True); continue
            steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
            if not steps_ck:
                print(f"  {dims} N={N}: aucun checkpoint, skip", flush=True); continue
            print(f"\n##### {dims} N={N} ({run_dir}) — checkpoints {steps_ck} #####", flush=True)

            # val-loss propre par checkpoint -> meilleur (val = 50 ép. figées)
            val_by_step = {}
            for s in steps_ck:
                ckpt = str(ckpt_root / f"{s:06d}" / "pretrained_model")
                policy = load_minicnn_policy(ckpt, device,
                    config_template=f"results/runs/lift/51_unet_d{dtag}/checkpoints/006000/pretrained_model")
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

            # rollout @ STEPS pas sur les 500 départs figés
            ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
            policy = load_minicnn_policy(ckpt, device,
                    config_template=f"results/runs/lift/51_unet_d{dtag}/checkpoints/006000/pretrained_model")
            pre, post = procs(ckpt)
            per_ep, agg = lift_eval.rollout_eval_chunked(policy, pre, post, states,
                device=device, num_inference_steps=STEPS, stop_on_success=True)
            del policy
            k = sum(e["success"] for e in per_ep)
            lo, hi = wilson_ci(k, len(per_ep))
            rec = {"down_dims": dims, "n_demos": N, "best_step": best, "val_loss": val_by_step[best],
                   "steps": STEPS, "n": len(per_ep), "n_success": k, "success_rate": agg["success_rate"],
                   "ci95_low": lo, "ci95_high": hi, "t_success_median": agg["t_success_median"],
                   "elapsed_s": agg["elapsed_s"]}
            results.append(rec)
            print(f"  => {dims} N={N} : {k}/{len(per_ep)} = {agg['success_rate']:.1%} "
                  f"[{lo:.1%}, {hi:.1%}]  t_succ_med={agg['t_success_median']} "
                  f"(val {val_by_step[best]:.4f}, {agg['elapsed_s'] / 60:.1f} min)", flush=True)
            OUT.write_text(json.dumps({"steps": STEPS, "n_eval": int(states.shape[0]),
                                       "results": results}, indent=2))

    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
