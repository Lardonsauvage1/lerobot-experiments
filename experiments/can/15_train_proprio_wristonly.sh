#!/bin/bash
# Phase 5 (Can) — mono-cam WRIST UNIQUEMENT (pas agentview).
# Même archi que 02_proprio (ResNet18 + U-Net [64,128,256], 9D), juste avec la caméra
# wrist (robot0_eye_in_hand) au lieu de agentview. Expérience de contrôle pour isoler
# la fragilité OOD de la wrist quand elle est seule.
# Lancer : nohup bash experiments/can/15_train_proprio_wristonly.sh > results/logs/can/run_15_proprio_wristonly.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_wristonly \
  --dataset.root=data_cache/lerobot_can_ph_proprio_wristonly \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/15_proprio_wristonly \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[15_proprio_wristonly] TRAIN DONE"
