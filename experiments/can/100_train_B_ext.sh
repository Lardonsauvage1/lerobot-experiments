#!/bin/bash
# B — CONTINUATION du joint pré-entraîné : reprend le checkpoint 40k -> entraîne 40k de plus (=80k réel).
# LR par groupe CONSTANT (encodeur 1e-5 / U-Net 1e-4). Logs PERSISTANTS (results/logs, pas /tmp -> plus perdus).
# But : capturer loss/lr/grad (perdus la 1re fois) pour le 4-panneaux + voir si B progresse au-delà de 40k.
# Lancer (mac2) : caffeinate -i nohup bash experiments/can/100_train_B_ext.sh > /tmp/run_B_ext.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
SRC=results/runs/can/joint_r34_pretrained/checkpoints/040000/pretrained_model
mkdir -p results/logs/can

ENCODER_LR=1e-5 UNET_LR=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims="[128,256,512]" --policy.vision_backbone="resnet34" \
  --policy.pretrained_backbone_weights="IMAGENET1K_V1" --policy.use_group_norm=false \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --policy.pretrained_path=$SRC \
  --dataset.repo_id=local/can_ph_joint_birdview --dataset.root=data_cache/lerobot_can_ph_joint_birdview --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/joint_r34_pretrained_ext \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 \
  2>&1 | tee results/logs/can/run_B_ext.log      # LOG PERSISTANT (results/logs, jamais /tmp)
echo "[100] $(date '+%H:%M') TRAIN DONE B ext (40k->80k), logs dans results/logs/can/run_B_ext.log"
