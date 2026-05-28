#!/bin/bash
# Phase 5 (Can) — image + proprio avec 2 caméras (agentview + robot0_eye_in_hand).
# Lève l'ambiguïté de profondeur d'une seule vue (les échecs du modèle 02_proprio venaient
# en partie de là). État 9D, ResNet18 partagé sur les 2 caméras (use_separate=False par défaut).
# Lancer : nohup bash experiments/can/06_train_proprio_wrist.sh > results/logs/can/run_06_proprio_wrist.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_wrist \
  --dataset.root=data_cache/lerobot_can_ph_proprio_wrist \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/06_proprio_wrist \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[06_proprio_wrist] TRAIN DONE"
