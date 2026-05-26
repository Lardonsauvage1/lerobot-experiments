#!/bin/bash
# Lift image+proprio (9D, sans coords cube) — ESCALADE "le plus gros" si le 16M (73) est trop faible.
# ResNet18 + U-Net [256,512,1024] (~77M), 15k steps. Le plus gros qui s'entraîne fiablement sur Mac.
# Lancer : nohup bash experiments/lift/75_train_proprio_big.sh > results/logs/lift/run_75_proprio_big.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[256,512,1024]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/lift_ph_proprio \
  --dataset.root=data_cache/lerobot_lift_ph_proprio \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/lift/75_proprio_big \
  --batch_size=32 --steps=15000 --save_freq=3000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[75_proprio_big] TRAIN DONE"
