#!/bin/bash
# A — joint DELTA (chunk-wise) : réentraîne le joint avec action = delta articulaire au lieu d'absolu.
# Le seul test qui RÉSOUT le joint (litt: delta 89,6% vs absolu 69%). Même archi que joint_r34_bigunet
# (R34 from-scratch, U-Net [128,256,512], 2 encodeurs séparés), LR constant 1e-4, 40k steps.
# Pour mac2 (training only). Garde-fou disque intégré. En FICHIER.
# Lancer (mac2) : caffeinate -i nohup bash experiments/can/95_train_joint_delta.sh > /tmp/run_joint_delta.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_joint_delta_birdview \
  --dataset.root=data_cache/lerobot_can_ph_joint_delta_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/joint_r34_delta \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[95] $(date '+%H:%M') TRAIN DONE joint delta"
