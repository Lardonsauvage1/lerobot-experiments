#!/bin/bash
# Phase 5 (Can) — image + proprio SEULE (9D, sans coords canette) — test de transférabilité réel.
# Modèle doit VOIR la canette dans l'image au lieu qu'on lui souffle sa position.
# Vision = ResNet18, U-Net [64,128,256], état 9D, 10k steps, save_freq 2000.
# Lancer : nohup bash experiments/can/02_train_proprio.sh > results/logs/can/run_02_proprio.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio \
  --dataset.root=data_cache/lerobot_can_ph_proprio \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/02_proprio \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[02_proprio] TRAIN DONE"
