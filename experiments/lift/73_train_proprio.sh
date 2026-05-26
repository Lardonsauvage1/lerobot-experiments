#!/bin/bash
# Lift "image + proprio SEULE" (sans coords du cube) — test de transférabilité réel.
# Le modèle doit VOIR le cube (image) au lieu qu'on lui donne sa position.
# Vision = ResNet18 (capable de localiser le cube), U-Net [64,128,256], état 9D (proprio).
# Train 150 / val 50, val-loss trackée. Sortie results/runs/lift/73_proprio/.
# Lancer : nohup bash experiments/lift/73_train_proprio.sh > results/logs/lift/run_73_proprio.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/lift_ph_proprio \
  --dataset.root=data_cache/lerobot_lift_ph_proprio \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/lift/73_proprio \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[73_proprio] TRAIN DONE"
