#!/usr/bin/env bash
# Apple CARTÉSIEN COMBINÉ (classique + 2 corrections) sur gb10 — GROS config qui marchait :
# R18 + U-Net[512,1024,2048] + crop[112,112], 128px, action pose TCP 6D + gripper (7D), 15 Hz.
# LR const 1e-4 + EMA. EPS (train) passé via env (val = épisodes classiques tenus à l'écart).
set -u
RES=128
CROP=$((RES*7/8))   # 128 -> 112
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=${EPS:?"passer EPS=[...] (indices train ; val = le complément)"}
STEPS=${STEPS:-40000}
echo "[cart-comb $(date '+%H:%M')] R18+[512,1024,2048]+crop[$CROP], 128px cartésien combiné, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet18 --policy.down_dims=[512,1024,2048] \
  --policy.crop_shape=[$CROP,$CROP] --policy.crop_is_random=true \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.n_obs_steps=2 \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/apple_cart_combined_128 --dataset.root=data_cache/lerobot_apple_cart_combined_128 --dataset.episodes=$EPS \
  --output_dir=results/runs/real/apple_cart_combined_128 \
  --batch_size=64 --steps=$STEPS --save_freq=20000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/apple_cart_combined.log
echo "[cart-comb $(date '+%H:%M')] TRAIN DONE"
