#!/usr/bin/env bash
# WRISTCAP — bras A (BASELINE sans poignet) : agentview SEULE, R34 + gros U-Net [128,256,512],
# LR const 1e-4 + EMA. Sur gb10 (dataset IMAGES, torchcodec HS sur aarch64). Juge = rollouts sim.
# But : la cible à battre (agentview-seule à HAUTE capacité, jamais testée — run 02 était R18).
# Lancer sur gb10 (dans tmux) : bash ~/lerobot-experiments/experiments/can/96_wristcap_A_train_agentview.sh
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(150)))" | tr -d ' ')   # train 0-149 ; val-loss auto 150-199
STEPS=${STEPS:-30000}
echo "[wristcap-A $(date '+%H:%M')] agentview seule, R34+bigU-Net, const 1e-4+EMA, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=[128,256,512] \
  --policy.device=cuda --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_img --dataset.root=data_cache/lerobot_can_ph_proprio_img --dataset.episodes=$EPS \
  --output_dir=results/runs/can/wristcap_A_agentview \
  --batch_size=32 --steps=$STEPS --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/wristcap_A.log
echo "[wristcap-A $(date '+%H:%M')] TRAIN DONE"
