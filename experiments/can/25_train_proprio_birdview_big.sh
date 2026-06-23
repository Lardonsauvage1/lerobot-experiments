#!/bin/bash
# Phase 5 (Can) — birdview + gros U-Net (Option A).
# Clone du `16_proprio_birdview.sh` avec down_dims=[128,256,512] (×3 sur le U-Net).
# Vise : tester si combiner les wins de 14 (gros U-Net : +13 pt sur wrist) ET 16
# (birdview : +20 pt vs wrist) bat les 74.8% @500 de 16.
# ~40 M params : 2× ResNet18 (22.4 M) + U-Net [128,256,512] (~17 M) + projections.
# Lancer : nohup bash experiments/can/25_train_proprio_birdview_big.sh > results/logs/can/run_25_proprio_birdview_big.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/25_proprio_birdview_big \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[25_proprio_birdview_big] TRAIN DONE"
