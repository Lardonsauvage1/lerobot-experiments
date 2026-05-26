#!/bin/bash
# Phase 5 (Can) — baseline "signal d'abord" : mini-CNN + U-Net [64,128,256], 12k steps.
# But : confirmer que Can est apprenable + tester si l'archi compressée généralise depuis Lift.
# State Can = 12D (proprio + can_pos), image 96px agentview. Train 150 / val 50 (val-loss trackée).
# Lancer détaché : nohup bash experiments/can/01_train_baseline.sh > results/logs/can/run_01_baseline.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')   # train = démos 0..149 (val = 150..199)

$PY -u experiments/lift/61_train_minicnn.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph \
  --dataset.root=data_cache/lerobot_can_ph \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/01_baseline \
  --batch_size=32 --steps=12000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[01_baseline] TRAIN DONE"
