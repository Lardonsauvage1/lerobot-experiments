#!/bin/bash
# Attend la conversion en images -> transfère sur gb10 -> relance l'entraînement (dataset images) dans tmux.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
IMGROOT=data_cache/lerobot_apple_joint_224_clean_img
LOG=results/logs/real/gb10_img_pipeline.log
log(){ echo "[gb10img $(date '+%H:%M')] $*" | tee -a "$LOG"; }

log "attente conversion images..."
while pgrep -f 24_to_images.py >/dev/null; do sleep 20; done
grep -q "TERMINE" results/logs/real/to_images.log || { log "conversion sans TERMINE"; exit 1; }
log "conversion finie : $(du -sh $IMGROOT 2>/dev/null | cut -f1)"

log "transfert dataset images -> gb10"
ssh gb10 'mkdir -p ~/lerobot-experiments/data_cache'
rsync -az --delete "$IMGROOT/" gb10:'~/lerobot-experiments/data_cache/lerobot_apple_joint_224_clean_img/' && log "transfert OK" || { log "rsync ECHEC"; exit 1; }
rsync -az experiments/real/23b_train_gb10_img.sh gb10:'~/lerobot-experiments/experiments/real/'

log "relance entraînement gb10 (images) dans tmux"
ssh gb10 'rm -rf ~/lerobot-experiments/results/runs/real/apple_clean_gb10 ~/lerobot-experiments/results/runs/real/apple_clean_gb10_ema 2>/dev/null
~/ffmpeg-env/bin/tmux kill-session -t train 2>/dev/null
~/ffmpeg-env/bin/tmux new-session -d -s train "bash ~/lerobot-experiments/experiments/real/23b_train_gb10_img.sh"
sleep 3; ~/ffmpeg-env/bin/tmux ls' | tee -a "$LOG"
log "entraînement gb10 (images) lancé. log /tmp/train_gb10.log"
