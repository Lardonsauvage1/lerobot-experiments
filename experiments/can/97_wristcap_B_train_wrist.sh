#!/usr/bin/env bash
# WRISTCAP — bras B (TEST avec poignet) : agentview + robot0_eye_in_hand, ENCODEURS SÉPARÉS,
# R34 + gros U-Net [128,256,512], LR const 1e-4 + EMA. Sur gb10 (dataset IMAGES). Juge = rollouts sim.
# But : le poignet aide-t-il ENFIN quand la capacité n'est plus le goulot ? (runs wrist passés = R18)
# Doit BATTRE le bras A (agentview seule même capacité) pour valider.
# Lancer sur gb10 (dans tmux) : bash ~/lerobot-experiments/experiments/can/97_wristcap_B_train_wrist.sh
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(150)))" | tr -d ' ')   # train 0-149 ; val-loss auto 150-199
STEPS=${STEPS:-30000}
echo "[wristcap-B $(date '+%H:%M')] agentview+poignet séparés, R34+bigU-Net, const 1e-4+EMA, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=[128,256,512] \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=cuda --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_wrist_img --dataset.root=data_cache/lerobot_can_ph_proprio_wrist_img --dataset.episodes=$EPS \
  --output_dir=results/runs/can/wristcap_B_wrist \
  --batch_size=32 --steps=$STEPS --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/wristcap_B.log
echo "[wristcap-B $(date '+%H:%M')] TRAIN DONE"
