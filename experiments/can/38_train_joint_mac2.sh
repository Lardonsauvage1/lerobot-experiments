#!/bin/bash
# Jumeau ARTICULAIRE de run 31, exécuté SUR mac2 (user sammurawka, cd via $HOME).
# = R34 + gros U-Net [128,256,512] + encodeurs séparés, birdview, MAIS action/obs en JOINTS.
# LR constant 1e-4 (patch CONST_LR de 50_train_valloss.py), batch 16, 40k, ckpt/2000.
# 150 premières démos (comme run 31) pour comparabilité.
# Lancé par le waiter depuis le Mac principal : ssh mac2 'nohup bash experiments/can/38_train_joint_mac2.sh > /tmp/run_joint.log 2>&1 &'

cd "$HOME/lerobot-experiments" || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_joint_birdview \
  --dataset.root=data_cache/lerobot_can_ph_joint_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/joint_r34_bigunet \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[joint_mac2] TRAIN DONE"
