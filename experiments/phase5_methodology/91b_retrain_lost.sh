#!/bin/bash
# mac2 : ré-entraîne les cooldowns mini PERDUS pendant le disque-plein (25,30,35,40k).
# Attend que la passe principale (91) finisse, puis entraîne + GARDE les checkpoints (pas de suppression).
# Lancer (mac2) : caffeinate -i nohup bash experiments/phase5_methodology/91b_retrain_lost.sh > /tmp/mini_cd_retrain.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
SRC=results/runs/phase5_methodology/mini_constant_all
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

echo "[91b] $(date '+%H:%M') attente fin passe principale (91)..."
while pgrep -f "61_train_minicnn" >/dev/null; do sleep 120; done
echo "[91b] $(date '+%H:%M') -> ré-entraînement des cooldowns perdus 25/30/35/40k"

for K in 25 30 35 40; do
  START=$((K*1000)); STEPS=$((START+5000)); D=$(printf '%06d' $STEPS)
  OUT=results/runs/phase5_methodology/mini_cd_${K}k
  [ -s "$OUT/checkpoints/$D/pretrained_model/model.safetensors" ] && { echo "[91b] ${K}k déjà présent"; continue; }
  INIT=$SRC/checkpoints/$(printf '%06d' $START)/pretrained_model
  [ -s "$INIT/model.safetensors" ] || { echo "[91b] source ${K}k absente"; continue; }
  echo "[91b] $(date '+%H:%M') COOLDOWN mini ${K}k (LR 1e-4->0 sur 5k)"
  MINI_INIT_CKPT=$INIT MINI_SCHED=cooldown MINI_START_STEP=$START $PY -u experiments/lift/61_train_minicnn.py \
    --policy.type=diffusion --policy.down_dims="[32,64,128]" --policy.vision_backbone=resnet18 \
    --policy.use_separate_rgb_encoder_per_camera=false --policy.crop_shape=null --policy.push_to_hub=false --policy.device=mps \
    --dataset.repo_id=local/can_ph_proprio_birdview --dataset.root=data_cache/lerobot_can_ph_proprio_birdview --dataset.episodes="$EPS" \
    --output_dir=$OUT --batch_size=32 --steps=$STEPS --save_freq=5000 --log_freq=200 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=0
done
echo "[91b] $(date '+%H:%M') DONE — 25/30/35/40k prêts (à évaluer par 90 au prochain run)"
