#!/usr/bin/env bash
# JOINT sim, agentview 96, BON CONFIG (recette gagnante) : R18 + U-Net[512,1024,2048] + crop[84,84].
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(150)))" | tr -d ' ')
STEPS=${STEPS:-40000}
echo "[joint-B $(date '+%H:%M')] BON config: R18+[512,1024,2048]+crop, agentview 96, joint, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet18 --policy.down_dims=[512,1024,2048] \
  --policy.crop_shape=[84,84] --policy.crop_is_random=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/can_ph_joint_agent_96_img --dataset.root=data_cache/lerobot_can_ph_joint_agent_96_img --dataset.episodes=$EPS \
  --output_dir=results/runs/can/joint_agent_goodconfig \
  --batch_size=64 --steps=$STEPS --save_freq=20000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/joint_B.log
echo "[joint-B $(date '+%H:%M')] DONE"
