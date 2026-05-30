#!/bin/bash
# Phase 5 (Can) — CONTINUATION de l'entraînement 08 (2 cams sep, 55%) pour 10k steps de plus.
# Charge les poids du checkpoint 010000 (best val-loss), repart sur 10k steps avec optimizer
# frais. But : voir si plus d'entraînement améliore le 55% rollout (la train loss était
# encore en lente descente à 10k).
# Lancer : nohup bash experiments/can/13_train_proprio_wrist_sep_ext.sh > results/logs/can/run_13_proprio_wrist_sep_ext.log 2>&1 &

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
  --policy.pretrained_path=results/runs/can/08_proprio_wrist_sep/checkpoints/010000/pretrained_model \
  --dataset.repo_id=local/can_ph_proprio_wrist \
  --dataset.root=data_cache/lerobot_can_ph_proprio_wrist \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/08_proprio_wrist_sep_ext \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[13_proprio_wrist_sep_ext] TRAIN DONE"
