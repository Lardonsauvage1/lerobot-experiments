#!/bin/bash
# Chain post-train Mac : 04 rollout 50 → 05 val_loss multi → 06 variance pure.
# Lancer : nohup bash experiments/phase5_methodology/post_train_chain.sh > results/logs/phase5_methodology/run_post_train.log 2>&1 &
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

echo "[chain] $(date '+%H:%M:%S') START — 04 rollout 50 d'abord (~12 h)"
$PY -u experiments/phase5_methodology/04_eval_50_mac.py || echo "[chain] !!! 04 ÉCHOUÉ"

echo "[chain] $(date '+%H:%M:%S') 04 DONE — 05 val_loss multi (~1 h)"
$PY -u experiments/phase5_methodology/05_val_loss_multi.py || echo "[chain] !!! 05 ÉCHOUÉ"

echo "[chain] $(date '+%H:%M:%S') 05 DONE — 06 variance pure (~2 h)"
$PY -u experiments/phase5_methodology/06_variance_pure.py || echo "[chain] !!! 06 ÉCHOUÉ"

echo "[chain] $(date '+%H:%M:%S') ALL POST-TRAIN DONE"
