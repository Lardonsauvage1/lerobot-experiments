#!/bin/bash
# Phase 5 (Can) — image+proprio 2 cams AVEC encodeurs vision SÉPARÉS (1 ResNet18 par caméra).
# Hypothèse : le shared encoder (06_proprio_wrist) dilue la capacité quand les 2 vues sont
# très différentes (agentview large vs wrist plongée). Avec un ResNet18 dédié par caméra,
# chacun se spécialise -> meilleurs features -> meilleur succès attendu.
# Coût : ~22 M params (vs 17 M shared), training ~1,4× plus lent côté vision.
# Lancer : nohup bash experiments/can/08_train_proprio_wrist_sep.sh > results/logs/can/run_08_proprio_wrist_sep.log 2>&1 &

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
  --dataset.repo_id=local/can_ph_proprio_wrist \
  --dataset.root=data_cache/lerobot_can_ph_proprio_wrist \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/08_proprio_wrist_sep \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[08_proprio_wrist_sep] TRAIN DONE"
