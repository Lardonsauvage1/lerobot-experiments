#!/bin/bash
# mac2 : entraîne les cooldowns mini-CNN CONSTANT, grille 5k (0-100k). Garde localement ; Principal (90) tire+évalue.
# Attend la fin de ses cooldowns joint (89). En FICHIER.
# Lancer (mac2) : caffeinate -i nohup bash experiments/phase5_methodology/91_minicnn_cooldown_mac2.sh > /tmp/mini_cd_mac2.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
SRC=results/runs/phase5_methodology/mini_constant_all
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

echo "[91] $(date '+%H:%M') attente fin cooldowns joint mac2 (89)..."
while pgrep -f "89_joint_dense|50_train_valloss.*joint_cd" >/dev/null; do sleep 120; done
echo "[91] $(date '+%H:%M') -> cooldowns mini-CNN (grille 5k 0-100k)"

for K in 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 5; do  # 10,15,20 déjà évalués -> skip ; 5 en dernier (peu utile)
  START=$((K*1000)); STEPS=$((K*1000+5000)); D=$(printf '%06d' $STEPS)
  OUT=results/runs/phase5_methodology/mini_cd_${K}k
  [ -s "$OUT/checkpoints/$D/pretrained_model/model.safetensors" ] && { echo "[91] ${K}k déjà fait"; continue; }
  INIT=$SRC/checkpoints/$(printf '%06d' $START)/pretrained_model
  [ -s "$INIT/model.safetensors" ] || { echo "[91] source ${K}k absente"; continue; }
  echo "[91] $(date '+%H:%M') COOLDOWN mini ${K}k (LR 1e-4->0 sur 5k)"
  MINI_INIT_CKPT=$INIT MINI_SCHED=cooldown MINI_START_STEP=$START $PY -u experiments/lift/61_train_minicnn.py \
    --policy.type=diffusion --policy.down_dims="[32,64,128]" --policy.vision_backbone=resnet18 \
    --policy.use_separate_rgb_encoder_per_camera=false --policy.crop_shape=null --policy.push_to_hub=false --policy.device=mps \
    --dataset.repo_id=local/can_ph_proprio_birdview --dataset.root=data_cache/lerobot_can_ph_proprio_birdview --dataset.episodes="$EPS" \
    --output_dir=$OUT --batch_size=32 --steps=$STEPS --save_freq=5000 --log_freq=200 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=0
  echo "[91] $(date '+%H:%M') ${K}k prêt"
done
echo "[91] $(date '+%H:%M') DONE"
