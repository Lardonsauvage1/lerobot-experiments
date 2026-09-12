#!/usr/bin/env bash
set -u
cd ~/lerobot-experiments
D=experiments/real
for RES in 96 128; do
  echo "[b2-all $(date '+%m-%d %H:%M')] === TRAIN ${RES}px ==="
  bash $D/32_train_b2_gb10.sh $RES
  echo "[b2-all $(date '+%m-%d %H:%M')] === COOLDOWN ${RES}px ==="
  bash $D/33_cooldown_b2_gb10.sh $RES
done
echo "[b2-all $(date '+%m-%d %H:%M')] === TOUT FINI (b2 96+128) ==="
