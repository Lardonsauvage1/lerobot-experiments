#!/usr/bin/env bash
# Apple batch2 (1 caméra FIXE) sur gb10. Arg = résolution (96 ou 128).
# R34 + U-Net[128,256,512], joint 6D absolu, LR const 1e-4 + EMA, PAS de crop. Dataset images.
set -u
RES=$1
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(70)))" | tr -d ' ')   # train 0-69, val-loss auto 70-78
STEPS=${STEPS:-30000}
echo "[apple-b2-$RES $(date '+%H:%M')] 1cam fixe ${RES}px, R34+[128,256,512], const+EMA, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=[128,256,512] \
  --policy.device=cuda --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/apple_b2_fixed_$RES --dataset.root=data_cache/lerobot_apple_b2_fixed_$RES --dataset.episodes=$EPS \
  --output_dir=results/runs/real/apple_b2_fixed_$RES \
  --batch_size=64 --steps=$STEPS --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/apple_b2_$RES.log
echo "[apple-b2-$RES $(date '+%H:%M')] TRAIN DONE"
