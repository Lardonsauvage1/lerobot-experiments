#!/bin/bash
# Enchaîne : attend la conversion nettoyée -> transfère le dataset propre sur atomman -> lance
# l'entraînement SUR atomman (CPU, config IDENTIQUE, save tous les 2000). Monitorisé côté Mac.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
DSROOT=data_cache/lerobot_apple_joint_224_clean
CLOG=results/logs/real/convert_clean.log
LOG=results/logs/real/clean_pipeline.log
mkdir -p results/logs/real
log(){ echo "[cleanpipe $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

log "attente fin conversion nettoyée..."
while pgrep -f 20_convert_clean.py >/dev/null; do sleep 30; done
grep -q "TERMINE" "$CLOG" || { log "ERREUR conversion sans TERMINE"; exit 1; }
NEP=$($PY -c "import json;print(json.load(open('$DSROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
NF=$($PY -c "import json;print(json.load(open('$DSROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0)
log "dataset propre : $NEP épisodes, $NF frames"
[ "$NEP" -lt 40 ] && { log "ERREUR <40 ép"; exit 1; }

log "transfert dataset propre -> atomman"
ssh atomman 'mkdir -p ~/lerobot-experiments/data_cache'
rsync -az --delete "$DSROOT/" atomman:'~/lerobot-experiments/data_cache/lerobot_apple_joint_224_clean/' && log "transfert OK" || { log "ERREUR rsync"; exit 1; }

log "lancement entraînement SUR atomman (CPU, config identique, save/2000, steps 30000)"
ssh atomman 'bash -lc "
cd ~/lerobot-experiments
EPS=\$(venv/bin/python -c \"print(list(range(40)))\" | tr -d \" \")
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 nohup taskset -c 0-11 venv/bin/python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims=[128,256,512] --policy.vision_backbone=resnet34 \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=cpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/apple_joint_224_clean --dataset.root=data_cache/lerobot_apple_joint_224_clean --dataset.episodes=\$EPS \
  --dataset.video_backend=pyav \
  --output_dir=results/runs/real/apple_clean_r34 --batch_size=16 --steps=30000 --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=4 > /tmp/train_clean.log 2>&1 &
echo LANCE PID \$!
"' | tee -a "$LOG"
log "entraînement lancé sur atomman (log distant /tmp/train_clean.log). Val-loss auto sur ép. 40-45."
