#!/bin/bash
# Phase 5 — Train dense pour étude méthodologique.
# 26 ResNet34 + birdview from scratch, 30 000 steps, save toutes les 100 steps.
# Utilise 50_train_valloss.py existant qui logue val_loss live à chaque log_freq.
# Lancer : nohup bash experiments/phase5_methodology/01_train_dense.sh > results/logs/phase5_methodology/run_01_train_dense.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
OUT="results/runs/phase5_methodology/26_resnet34_dense"

if [ -d "$OUT/checkpoints" ]; then
  echo "[01] $(date '+%H:%M:%S') $OUT existe déjà, refuse de écraser. Supprime-le avant de relancer."
  exit 1
fi

echo "[01] $(date '+%H:%M:%S') TRAIN dense 30k steps, save 100, log 100 -> $OUT"
$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir="$OUT" \
  --batch_size=32 --steps=30000 --save_freq=100 --log_freq=100 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[01] $(date '+%H:%M:%S') TRAIN DONE"
