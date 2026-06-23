#!/bin/bash
# Phase 5 Can — variante de `16` (2× ResNet18 sep + birdview = 74.8% @500) avec
# ResNet34 au lieu de ResNet18 (vision ×2 capacité, ~21 M par cam vs ~11 M).
# Iso single-change vs 16 : SEUL le backbone vision change. Tout le reste identique
# (U-Net [64,128,256], birdview, 9D, 10 000 steps, batch 32). Permet d'isoler
# l'effet de la capacité vision sur la perf vision-pure Can.
# Total ≈ 47 M params (42 M vision + 5 M U-Net + projections).
# Lancer : nohup bash experiments/can/26_train_proprio_birdview_resnet34.sh > results/logs/can/run_26_proprio_birdview_resnet34.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

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
  --output_dir=results/runs/can/26_proprio_birdview_resnet34 \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[26_proprio_birdview_resnet34] TRAIN DONE"
