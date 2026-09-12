#!/usr/bin/env bash
# WRISTCAP-STD — bras A" (BASELINE) : agentview seule, RECETTE STANDARD Diffusion Policy robomimic :
# 84px, ResNet18, U-Net [512,1024,2048], crop 76<-84, batch 64, 196 démos, entraînement LONG. LR const+EMA. gb10.
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(196)))" | tr -d ' ')
STEPS=${STEPS:-40000}
echo "[wristcap84-A $(date '+%H:%M')] STD agentview 84, R18, U-Net[512,1024,2048], crop76, batch64, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet18 --policy.down_dims=[512,1024,2048] \
  --policy.crop_shape=[76,76] --policy.crop_is_random=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/can_ph_proprio_84_img --dataset.root=data_cache/lerobot_can_ph_proprio_84_img --dataset.episodes=$EPS \
  --output_dir=results/runs/can/wristcap84_A_agentview \
  --batch_size=64 --steps=$STEPS --save_freq=20000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/wristcap84_A.log
echo "[wristcap84-A $(date '+%H:%M')] TRAIN DONE"
