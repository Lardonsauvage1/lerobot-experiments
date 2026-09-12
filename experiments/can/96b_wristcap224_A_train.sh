#!/usr/bin/env bash
# WRISTCAP+224+AUG — bras A' (BASELINE) : agentview SEULE, 224px, R34 + gros U-Net,
# AVEC AUGMENTATION (random crop 196<-224 + color jitter). LR const 1e-4 + EMA. gb10 (images).
# But : cible à battre, à la config "comme les autres" (haute res + augmentation).
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(150)))" | tr -d ' ')
STEPS=${STEPS:-30000}
echo "[wristcap224-A $(date '+%H:%M')] agentview 224, R34+bigU-Net, crop196+jitter, const+EMA, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=[128,256,512] \
  --policy.crop_shape=[196,196] --policy.crop_is_random=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/can_ph_proprio_224_img --dataset.root=data_cache/lerobot_can_ph_proprio_224_img --dataset.episodes=$EPS \
  --output_dir=results/runs/can/wristcap224_A_agentview \
  --batch_size=16 --steps=$STEPS --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/wristcap224_A.log
echo "[wristcap224-A $(date '+%H:%M')] TRAIN DONE"
