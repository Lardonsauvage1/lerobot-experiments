#!/usr/bin/env bash
set -u
cd ~/lerobot-experiments
D=experiments/real
for RES in 96 128; do
  echo "[cart-all $(date '+%m-%d %H:%M')] === TRAIN ${RES}px ==="; bash $D/36_train_cart_gb10.sh $RES
  echo "[cart-all $(date '+%m-%d %H:%M')] === COOLDOWN ${RES}px ==="; bash $D/37_cooldown_cart_gb10.sh $RES
done
echo "[cart-all $(date '+%m-%d %H:%M')] === TOUT FINI (cart 96+128) ==="
