#!/bin/bash
# Joint PRÉ-ENTRAÎNÉ ImageNet en RÉSOLUTION 224 (native ImageNet).
# Combine : backbone IMAGENET1K_V1 (BN gardé) + LR par groupe constant (encodeur 1e-5 / U-Net 1e-4)
# + images 224x224 (au lieu de 96). Même archi joint (R34, U-Net [128,256,512], 2 encodeurs).
# LR CONSTANT (méthode validée) -> cooldown appliqué ensuite. En FICHIER.
# Lancer : caffeinate -i nohup bash experiments/can/96_train_joint_pretrained_224.sh > /tmp/run_joint_pt224.log 2>&1 &
set -u
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
  --dataset.repo_id=local/can_ph_joint_birdview_224 \
  --dataset.root=data_cache/lerobot_can_ph_joint_birdview_224 \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/joint_r34_pretrained_224 \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[96] $(date '+%H:%M') TRAIN DONE joint pré-entraîné 224"
