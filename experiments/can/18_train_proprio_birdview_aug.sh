#!/bin/bash
# Phase 5 (Can) — birdview MAIS avec augmentations d'image (crop + color/affine).
# Même archi que 16 (2 cams agentview+birdview séparés, U-Net [64,128,256], 9D)
# mais avec :
#   - crop_shape=[84,84] : random crop 84x84 (de 96x96) à l'entraînement, center crop à l'éval
#   - image_transforms.enable=true : color jitter (brightness/contrast/sat/hue) + affine légère
# Objectif : robustesse OOD + +X pt de succès sur 500 rollouts.
# Lancer : nohup bash experiments/can/18_train_proprio_birdview_aug.sh > results/logs/can/run_18_proprio_birdview_aug.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape="[84,84]" \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --dataset.image_transforms.enable=true \
  --output_dir=results/runs/can/18_proprio_birdview_aug \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[18_proprio_birdview_aug] TRAIN DONE"
