# Run 46 — Diffusion Policy (Robomimic Lift) — 100 % success ⭐

## But
Première Diffusion Policy du projet sur Lift, via la recette officielle `lerobot-train`. Objectif : dépasser le plafond MLP (70 %, run 44) sur cette tâche multimodale.

## Modèle
- **Type** : Diffusion Policy LeRobot (U-Net 1D conditionnel + DDPM).
- **Vision** : ResNet18 (GroupNorm) + spatial-softmax 32 keypoints.
- **U-Net** : `down_dims [512, 1024, 2048]`, kernel 5, `diffusion_step_embed_dim 128`, FiLM.
- **I/O** : image 96×96×3 + état 19D ; `n_obs_steps=2`, `horizon=16`, `n_action_steps=8` ; action 7D.
- **Bruit** : DDPM, `num_train_timesteps=100` ; eval à `num_inference_steps=10`.
- **Params** : **263.7 M** (~1.05 Go fp32) — dont **95.8 % dans le U-Net**, 4.2 % vision.

## Dataset
- `local/lift_ph` (Robomimic Lift, proficient-human), **200 démos** / 9 666 frames, 20 fps.
- ⚠️ Entraîné sur les **200 démos** (pas de split val) → non auditable en généralisation. Gardé comme « preuve max » ; le baseline de la phase compression sera réentraîné sur 150 (voir `docs/COMPRESSION.md`).

## Entraînement
- `lerobot-train`, 15K steps prévus, **batch 32**, `num_workers 2`, seed 1000, save_freq 3000, wandb off.
- Évalué jusqu'au checkpoint **12K** (train loss ~0.037).

## Résultats (eval Robomimic env + sign fix, 20 ép./checkpoint, init states aléatoires)
| step | succès | avg max_z (marge de levage) |
|------|--------|------|
| 3000 | **100 %** | 0.918 |
| 6000 | **100 %** | 0.984 |
| 9000 | **100 %** | 0.978 |
| 12000 | **100 %** | 1.006 |

Succès saturé à 100 % dès 3K ; seule la marge de levage continue de progresser. ⚠️ Comparaison **biaisée** (init states différents par checkpoint). Voir `checkpoint_eval.json` + `checkpoint_comparison.png`.

## Bugs d'eval résolus (5)
DDPM 100 steps par défaut · `select_action` ne normalise pas (pipeline pre/post LeRobot v0.5) · init env hors distribution (reset depuis init state HDF5) · mismatch Robosuite 1.4.1↔1.5.2 sur `object-state` · sign flip `object[7:10]`. Détail : `README.md` § Run 46.

## Reproductibilité
```bash
venv312/bin/lerobot-train --policy.type=diffusion --policy.repo_id=local/lift_ph_diffusion \
  --policy.push_to_hub=false --dataset.repo_id=local/lift_ph --dataset.root=data_cache/lerobot_lift_ph \
  --output_dir=results/runs/lift/46_diffusion_official --steps=15000 --batch_size=32 \
  --eval_freq=999999 --save_freq=3000 --num_workers=2 --wandb.enable=false
```
Eval comparée : `python -u experiments/lift/46_eval_all_checkpoints.py`
