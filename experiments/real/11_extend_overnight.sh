#!/bin/bash
# PROLONGATION overnight du run "pomme" pour observer l'evolution de la loss (dynamique d'overfit).
# RESUME depuis le dernier checkpoint (2000) = etat exact (poids + optimiseur), LR const 1e-4 + EMA.
# config EN PLACE (sinon LeRobot cherche model.safetensors a cote du config). steps/log_freq en CLI.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/real/apple_joint_224_r34
CFG_RESUME=$RUN/checkpoints/last/pretrained_model/train_config.json
LOG=results/logs/real/extend.log
mkdir -p results/logs/real
[ -s "$CFG_RESUME" ] || { echo "config last absent"; exit 1; }

echo "[extend $(date '+%m-%d %H:%M')] RESUME depuis $(readlink $RUN/checkpoints/last) -> cible 50000, LR const 1e-4, EMA, log/200" | tee -a "$LOG"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG_RESUME --resume=true --steps=50000 --log_freq=200 >> "$LOG" 2>&1
echo "[extend $(date '+%m-%d %H:%M')] STOP (rc=$?)" | tee -a "$LOG"
