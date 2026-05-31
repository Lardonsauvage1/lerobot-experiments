#!/bin/bash
# Phase 5 (Can) — 2 cams séparés AVEC U-Net plus gros [128,256,512] (~40M params total).
# Hypothèse : le 08 ([64,128,256], 28M) plafonne à 55%. Plus de capacité U-Net pourrait
# aider à gérer le conditionnement multi-cam plus complexe.
# Lancer : nohup bash experiments/can/14_train_proprio_wrist_sep_big.sh > results/logs/can/run_14_proprio_wrist_sep_big.log 2>&1 &

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
  --dataset.repo_id=local/can_ph_proprio_wrist \
  --dataset.root=data_cache/lerobot_can_ph_proprio_wrist \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/14_proprio_wrist_sep_big \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[14_proprio_wrist_sep_big] TRAIN DONE"
