#!/usr/bin/env bash
# WRISTCAP+224+AUG maître : train A' -> cd A' -> train B' -> cd B' (séquentiel, gb10).
# tmux : ~/ffmpeg-env/bin/tmux new -d -s wc224 "bash ~/lerobot-experiments/experiments/can/99b_wristcap224_run_all.sh"
set -u
cd ~/lerobot-experiments
D=experiments/can
log(){ echo "[wc224-all $(date '+%m-%d %H:%M')] $*"; }

log "=== BRAS A' : agentview 224+aug ==="
bash $D/96b_wristcap224_A_train.sh
log "=== COOLDOWN A' ==="
bash $D/98_wristcap_cooldown.sh wristcap224_A_agentview

log "=== BRAS B' : agentview+poignet 224+aug ==="
bash $D/97b_wristcap224_B_train.sh
log "=== COOLDOWN B' ==="
bash $D/98_wristcap_cooldown.sh wristcap224_B_wrist

log "=== TOUT FINI (224+aug) : cooldowns prêts pour éval ==="
