#!/bin/bash
# Prolonge le training JOINT de +10k (40k -> 50k) sur mac2, SANS RIEN CHANGER (LR constant, même
# dataset/archi via le train_config sauvegardé). Reprend depuis checkpoints/last (optimiseur inclus).
# mac2 étant occupé par le look-at, on ATTEND sa fin avant de lancer (pas 2 trainings en même temps).
# Lancer (sur le Principal) : nohup bash experiments/can/45_resume_joint_plus10k.sh > /tmp/resume_joint10k.log 2>&1 &
set -u
RUN=results/runs/can/joint_r34_bigunet
PATL="50_train_valloss.*lookat"
PATJ="50_train_valloss.*joint_r34"

echo "[45] $(date '+%H:%M') attente de la fin du look-at sur mac2..."
while ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '$PATL' >/dev/null" 2>/dev/null; do
  sleep 120
done
echo "[45] $(date '+%H:%M') look-at terminé -> reprise JOINT +10k (steps=50000) sur mac2"

ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --config_path=$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true \
    --output_dir=$RUN --steps=50000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    >> /tmp/run_joint.log 2>&1 & echo joint_resume_PID \$!"
sleep 8
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PATJ' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 43200 >/dev/null 2>&1 & echo \"[45] joint repris PID \$P + caffeinate 12h\"; \
    grep -ao '[0-9]*/50000' /tmp/run_joint.log | tail -1; \
  else echo '[45] ⚠️ pas reparti :'; tail -8 /tmp/run_joint.log; fi"
echo "[45] DONE"
