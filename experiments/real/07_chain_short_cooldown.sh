#!/bin/bash
# Attend la fin du cooldown 5k (process 50_train_valloss termine + "COOLDOWN DONE"), puis lance le cooldown court 1k.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
CDLOG=results/logs/real/cooldown.log
CHAINLOG=results/logs/real/chain_short.log
log(){ echo "[chain $(date '+%m-%d %H:%M')] $*" | tee -a "$CHAINLOG"; }

log "attente fin du cooldown 5k..."
# tant qu'un train tourne, on attend (c'est le cooldown 5k)
while pgrep -f "50_train_valloss" >/dev/null; do sleep 60; done
if ! grep -q "COOLDOWN DONE" "$CDLOG"; then
  log "ATTENTION : plus de process mais pas de 'COOLDOWN DONE' dans $CDLOG (crash ?). Je lance quand meme le court depuis ckpt 2000 (indépendant)."
fi
log "cooldown 5k termine -> lancement du cooldown court 1k"
caffeinate -i bash experiments/real/06_cooldown_2000_short1k.sh
log "chain terminee."
