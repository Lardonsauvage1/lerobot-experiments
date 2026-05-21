# Run 47 — Baseline 150 démos (référence du sweep compression)

## But
Établir le baseline pleine-taille **sous le protocole Phase 0** (train 150 / val 50, métriques continues + val-loss), pour servir de référence propre au sweep U-Net. Le run 46 (200 démos) reste « preuve max » mais hors-comparaison.

## Setup
- Diffusion Policy LeRobot, archi **identique au run 46** (ResNet18 + U-Net `down_dims [512,1024,2048]`, 263.7 M params).
- Dataset : `local/lift_ph`, **train = démos 0–149** (split contigu, cf. `phase4_split.json`), val = 150–199.
- `lerobot-train`, **12000 steps**, batch 32, save_freq 1500 (8 checkpoints), log_freq 50, seed 42, device mps. ~11h26 sur MPS.
- Train loss : 0.067 (1.5K) → **0.029** (12K), pas d'instabilité.

## Résultat clé : 150 démos suffisent, et la val-loss donne le point d'arrêt
Éval sur les 50 démos val held-out (`49_phase0_curve.py`) :

| step | succ | t_succ | max_z | hold | val_loss |
|------|------|--------|-------|------|----------|
| 1500 | 96 % | 47.0 | 0.944 | 0.39 | 0.0762 |
| 3000 | 96 % | 45.5 | 0.942 | 0.61 | 0.0693 |
| 4500 | 100 % | 44.0 | 0.984 | 0.60 | 0.0664 |
| **6000** | **100 %** | **43.0** | **1.033** | 0.48 | **0.0649 ← min** |
| 7500 | 100 % | 44.0 | 1.027 | 0.38 | 0.0659 |
| 9000 | 100 % | 43.0 | 0.997 | 0.45 | 0.0831 |
| 10500 | 100 % | 43.0 | 1.015 | 0.31 | 0.0966 |
| 12000 | 100 % | 43.0 | 1.026 | 0.32 | 0.1001 |

- **100 % de succès dès 4500 steps** → 150 démos suffisent, généralisation OK.
- **Val-loss minimale à 6000 puis remonte** (+54 % à 12K) = overfit classique ; le hold (maintien) se dégrade en parallèle. Le succès binaire (plat 100 %) ne montrait rien.
- **Point de fonctionnement retenu : checkpoint 6000** (succès 100 %, val-loss min, max_z pic, t_success au mieux). Entraîner jusqu'à 12K = calcul gaspillé.

→ Pour le sweep U-Net : entraîner ~6000 steps avec `50_train_valloss.py` (val-loss continue) et early-stop au minimum.

## Fichiers
`loss_curves.png` (train), `phase0_curve.{json,png}` (rollout + val vs steps), checkpoints locaux (gitignorés).

## Repro
```bash
venv312/bin/lerobot-train --policy.type=diffusion --policy.repo_id=local/lift_ph_diffusion \
  --policy.push_to_hub=false --policy.device=mps --dataset.repo_id=local/lift_ph \
  --dataset.root=data_cache/lerobot_lift_ph \
  --dataset.episodes="$(cat results/runs/lift/phase4_train_episodes.txt)" \
  --output_dir=results/runs/lift/47_baseline_150 --steps=12000 --batch_size=32 \
  --save_freq=1500 --eval_freq=999999 --log_freq=50 --num_workers=2 --wandb.enable=false --seed=42
python -u experiments/lift/49_phase0_curve.py --run-dir results/runs/lift/47_baseline_150 --episodes 50
```
