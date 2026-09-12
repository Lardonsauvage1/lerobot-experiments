#!/usr/bin/env bash
# Entraînement "propre" SUR gb10 (GPU Blackwell). À lancer dans tmux :
#   tmux new -s train 'bash ~/lerobot-experiments/experiments/real/23_train_gb10.sh'
#   (détacher : Ctrl-b puis d  |  reprendre : tmux attach -t train)
# Config IDENTIQUE (R34+U-Net[128,256,512], 224px, batch16, LR const 1e-4, EMA), save tous les 2000.
set -u
cd ~/lerobot-experiments
. venv/bin/activate
# env qui débloque torchcodec : FFmpeg user-space + toutes les libs NVIDIA + NPP
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(40)))" | tr -d " ")   # train 0-39 ; val-loss auto 40-45
echo "[gb10 $(date '+%H:%M')] TRAIN 30k, batch16, GPU (torchcodec), save/2000 -> apple_clean_gb10"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims=[128,256,512] --policy.vision_backbone=resnet34 \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=cuda --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/apple_joint_224_clean --dataset.root=data_cache/lerobot_apple_joint_224_clean --dataset.episodes=$EPS \
  --dataset.video_backend=torchcodec \
  --output_dir=results/runs/real/apple_clean_gb10 --batch_size=16 --steps=30000 --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/train_gb10.log
echo "[gb10 $(date '+%H:%M')] TRAIN DONE"
