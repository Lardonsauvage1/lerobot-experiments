#!/bin/bash
# B (test #2 « extraire le max d'une archi ») : joint ResNet34 PRÉ-ENTRAÎNÉ ImageNet.
# Backbone IMAGENET1K_V1 + BatchNorm gardé (use_group_norm=false, obligatoire si pré-entraîné)
# + LR PAR GROUPE (encodeur vision 1e-5 / U-Net 1e-4, le 10:1 enfin justifié pour un encodeur pré-entraîné).
# From scratch (U-Net random), LR constant par groupe. Pour mac2 (training only). En FICHIER.
# Lancer (mac2) : caffeinate -i nohup bash experiments/can/81_train_joint_pretrained.sh > /tmp/run_joint_pretrained.log 2>&1 &
set -e
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

ENCODER_LR=1e-5 UNET_LR=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.vision_backbone="resnet34" \
  --policy.pretrained_backbone_weights="IMAGENET1K_V1" \
  --policy.use_group_norm=false \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_joint_birdview \
  --dataset.root=data_cache/lerobot_can_ph_joint_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/joint_r34_pretrained \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[81] DONE pretrained"
