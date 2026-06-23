#!/bin/bash
# Phase 5 (Can) — birdview 2 cams séparés MAIS à RÉSOLUTION 160 px (vs 96 standard).
# Hypothèse : à 96 px la canette ne fait que 6-10 px de large -> orientation/position
# imprécises. À 160 px la canette fait 10-16 px -> meilleure perception spatiale.
# Pas d'aug (crop ni image_transforms) car démasqués comme nocifs à 96.
# Lancer : nohup bash experiments/can/23_train_proprio_birdview_160.sh > results/logs/can/run_23_proprio_birdview_160.log 2>&1 &

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
  --dataset.repo_id=local/can_ph_proprio_birdview_160 \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview_160 \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/23_proprio_birdview_160 \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[23_proprio_birdview_160] TRAIN DONE"
