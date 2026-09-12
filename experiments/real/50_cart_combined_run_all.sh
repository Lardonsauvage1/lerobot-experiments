#!/usr/bin/env bash
# Orchestrateur cartésien combiné (gb10) : train -> cooldown. EPS via env.
set -u
cd ~/lerobot-experiments
echo "===== [CART-COMB] TRAIN $(date) ====="
EPS=${EPS:?"EPS requis"} bash experiments/real/48_train_cart_combined.sh || { echo "TRAIN ECHEC"; exit 1; }
echo "===== [CART-COMB] COOLDOWN $(date) ====="
bash experiments/real/49_cooldown_cart_combined.sh || { echo "COOLDOWN ECHEC"; exit 1; }
echo "===== [CART-COMB] TOUT FINI $(date) ====="
