#!/bin/bash
# Phase 5 (Can) — birdview + CROP SEUL (ablation, sans color jitter ni affine).
# Isole l'effet du random crop 84x84 (au train) / center crop (à l'éval).
# Si rollout ≈ 75 %  -> coupable = color/affine ; si reste ≈ 30 % -> coupable = crop train/eval.
# Lancer : nohup bash experiments/can/21_train_proprio_birdview_crop.sh > results/logs/can/run_21_proprio_birdview_crop.log 2>&1 &

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
  --output_dir=results/runs/can/21_proprio_birdview_crop \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[21_proprio_birdview_crop] TRAIN DONE"
