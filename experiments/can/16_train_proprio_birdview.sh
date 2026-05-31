#!/bin/bash
# Phase 5 (Can) — 2 cams séparés AVEC agentview + BIRDVIEW (au lieu de wrist).
# Hypothèse : la birdview (vue de dessus, scene-fixée) apporte de la parallaxe avec
# agentview SANS le problème OOD de la wrist. Devrait battre wrist+2cams séparés (58%)
# et possiblement battre le mono-cam (72%).
# Lancer : nohup bash experiments/can/16_train_proprio_birdview.sh > results/logs/can/run_16_proprio_birdview.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/16_proprio_birdview \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[16_proprio_birdview] TRAIN DONE"
