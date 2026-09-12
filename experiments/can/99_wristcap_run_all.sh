#!/usr/bin/env bash
# WRISTCAP maître : enchaîne les 2 bras EN SÉQUENCE (évite la contention GPU) :
#   train A -> cooldown A -> train B -> cooldown B.
# Lancer sur gb10 dans tmux : ~/ffmpeg-env/bin/tmux new -d -s wristcap "bash ~/lerobot-experiments/experiments/can/99_wristcap_run_all.sh"
set -u
cd ~/lerobot-experiments
D=experiments/can
log(){ echo "[wristcap-all $(date '+%m-%d %H:%M')] $*"; }

log "=== BRAS A : agentview seule ==="
bash $D/96_wristcap_A_train_agentview.sh
log "=== COOLDOWN A ==="
bash $D/98_wristcap_cooldown.sh wristcap_A_agentview

log "=== BRAS B : agentview + poignet ==="
bash $D/97_wristcap_B_train_wrist.sh
log "=== COOLDOWN B ==="
bash $D/98_wristcap_cooldown.sh wristcap_B_wrist

log "=== TOUT FINI : cooldowns prêts pour éval ==="
log "A: results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model"
log "B: results/runs/can/wristcap_B_wrist/cooldown/checkpoints/005000/pretrained_model"
