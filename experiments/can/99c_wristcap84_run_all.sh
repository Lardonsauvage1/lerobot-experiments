#!/usr/bin/env bash
# WRISTCAP-STD maître : train A" -> cd A" -> train B" -> cd B" (recette standard DP, 84px).
set -u
cd ~/lerobot-experiments
D=experiments/can
log(){ echo "[wc84-all $(date '+%m-%d %H:%M')] $*"; }
log "=== BRAS A\" : agentview 84 STD ==="
bash $D/96c_wristcap84_A_train.sh
log "=== COOLDOWN A\" ==="
bash $D/98_wristcap_cooldown.sh wristcap84_A_agentview
log "=== BRAS B\" : agentview+poignet 84 STD ==="
bash $D/97c_wristcap84_B_train.sh
log "=== COOLDOWN B\" ==="
bash $D/98_wristcap_cooldown.sh wristcap84_B_wrist
log "=== TOUT FINI (84 STD) ==="
