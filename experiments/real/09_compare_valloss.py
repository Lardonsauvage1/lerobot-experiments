#!/usr/bin/env python
"""Compare la VAL-LOSS des candidats "prise de pomme" pour choisir le modele deployable.

Val-loss = moyenne sur TOUT le val (ep. 40-45), meme preprocessor que le train, MODE eval,
moyennee sur N_SEEDS passes (le loss diffusion tire un timestep de bruit aleatoire -> on fixe
le seed et on moyenne pour un chiffre stable et comparable). Replique 50_train_valloss.

⚠️ Corps sous garde __main__ : avec num_workers>0 (macOS spawn) les workers DataLoader
re-importent le module ; sans garde ils ré-exécuteraient tout (spam + ecrasement du json).
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

RUN = Path("results/runs/real/apple_joint_224_r34")
VAL_EPISODES = [40, 41, 42, 43, 44, 45]
N_SEEDS = 5
NUM_WORKERS = 2   # >0 pour un decodage video parallelise (num_workers=0 = ultra lent)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CANDIDATES = {
    "deployable cooldown5k@2000":   RUN / "cooldown_002000/checkpoints/005000/pretrained_model",
    "cooldown5k@12000 brut":        RUN / "cooldown_012000/checkpoints/005000/pretrained_model",
    "cooldown5k@12000 EMA":         RUN / "cooldown_012000_ema/checkpoints/005000/pretrained_model",
}

cfg = dataset = stats = val_ds = None   # globals remplis par main()


def build_preprocessor(policy, path):
    kw = {"preprocessor_overrides": {
        "device_processor": {"device": DEVICE.type},
        "normalizer_processor": {
            "stats": stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }}
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(path), **kw)
    return pre


def eval_candidate(name, path):
    cfg.policy.pretrained_path = str(path)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval().to(DEVICE)
    pre = build_preprocessor(policy, path)
    per_seed = []
    for s in range(N_SEEDS):
        torch.manual_seed(1000 + s)
        loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        losses = []
        with torch.no_grad():
            for batch in loader:
                out = policy.forward(pre(batch))
                loss = out[0] if isinstance(out, (tuple, list)) else out
                losses.append(float(loss))
        per_seed.append(np.mean(losses))
    del policy
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return float(np.mean(per_seed)), float(np.std(per_seed))


def main():
    global cfg, dataset, stats, val_ds
    base_ckpt = RUN / "cooldown_002000/checkpoints/005000/pretrained_model"
    print(f"[compare] chargement cfg depuis {base_ckpt}  | device={DEVICE}")
    cfg = TrainPipelineConfig.from_pretrained(base_ckpt)
    cfg.dataset.episodes = list(range(40))
    dataset = make_dataset(cfg)
    stats = dataset.meta.stats
    val_ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root,
                            episodes=VAL_EPISODES, delta_timestamps=dataset.delta_timestamps)
    print(f"[compare] val: {len(VAL_EPISODES)} episodes, {val_ds.num_frames} frames")

    results = {}
    for name, path in CANDIDATES.items():
        if not (Path(path) / "model.safetensors").exists():
            print(f"[compare] SKIP {name} (absent)"); continue
        try:
            mean, std = eval_candidate(name, path)
            results[name] = (mean, std)
            print(f"[compare] {name:32s} val-loss = {mean:.5f} ± {std:.5f}")
        except Exception as e:
            print(f"[compare] ERREUR {name}: {e}")

    ranked = sorted(results.items(), key=lambda kv: kv[1][0])
    print("\n===== CLASSEMENT (val-loss croissante) =====")
    for i, (name, (m, sd)) in enumerate(ranked):
        star = "  <== DEPLOYABLE" if i == 0 else ""
        print(f"{i+1}. {name:32s} {m:.5f} ± {sd:.5f}{star}")

    out = {"device": DEVICE.type, "n_seeds": N_SEEDS, "val_episodes": VAL_EPISODES,
           "candidates": {k: {"path": str(CANDIDATES[k]), "val_loss": v[0], "std": v[1]} for k, v in results.items()},
           "ranking": [{"rank": i+1, "name": n, "val_loss": m, "std": sd} for i, (n, (m, sd)) in enumerate(ranked)],
           "deployable": ranked[0][0] if ranked else None,
           "deployable_path": str(CANDIDATES[ranked[0][0]]) if ranked else None}
    (RUN / "compare_valloss.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RUN/'compare_valloss.json'}")
    print(f"DEPLOYABLE = {out['deployable']}  ({out['deployable_path']})")


if __name__ == "__main__":
    main()
