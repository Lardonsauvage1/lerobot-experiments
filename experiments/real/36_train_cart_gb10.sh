#!/usr/bin/env bash
# Apple b2 CARTÉSIEN (1 cam fixe) sur gb10 — BONNE ARCHI (gagnante en sim) : R18 + U-Net[512,1024,2048] + crop.
# Arg = résolution (96 ou 128). Action = pose TCP 6D + gripper (7D). LR const 1e-4 + EMA. log détaillé.
set -u
RES=$1
CROP=$((RES*7/8))   # 0.875x : 96->84, 128->112
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(70)))" | tr -d ' ')   # train 0-69, val-loss 70-78
STEPS=${STEPS:-40000}
echo "[cart-$RES $(date '+%H:%M')] BONNE archi: R18+[512,1024,2048]+crop[$CROP], ${RES}px cartésien, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet18 --policy.down_dims=[512,1024,2048] \
  --policy.crop_shape=[$CROP,$CROP] --policy.crop_is_random=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/apple_b2cart_fixed_$RES --dataset.root=data_cache/lerobot_apple_b2cart_fixed_$RES --dataset.episodes=$EPS \
  --output_dir=results/runs/real/apple_b2cart_fixed_$RES \
  --batch_size=64 --steps=$STEPS --save_freq=20000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/apple_cart_$RES.log
echo "[cart-$RES $(date '+%H:%M')] TRAIN DONE"
