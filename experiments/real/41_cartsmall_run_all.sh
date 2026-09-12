#!/usr/bin/env bash
set -u
cd ~/lerobot-experiments
D=experiments/real
for RES in 96 128; do
  echo "[cartsmall-all $(date '+%m-%d %H:%M')] === TRAIN ${RES}px ==="; bash $D/39_train_cartsmall_gb10.sh $RES
  echo "[cartsmall-all $(date '+%m-%d %H:%M')] === COOLDOWN ${RES}px ==="; bash $D/40_cooldown_cartsmall_gb10.sh $RES
done
echo "[cartsmall-all $(date '+%m-%d %H:%M')] === TOUT FINI (cartsmall 96+128) ==="
