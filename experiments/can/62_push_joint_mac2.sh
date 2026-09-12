#!/bin/bash
# PUSH joint au-delà de 50k : resume 50k -> 80k (LR constant 1e-4) sur mac2, DÈS que le
# fine-tune prolongé (waiter 60) libère mac2. But : voir si le joint se stabilise / s'améliore
# bien au-delà de 50k (actuellement très bruité 12-56%). En FICHIER. last->050000 déjà sur mac2.
# Lancer : nohup bash experiments/can/62_push_joint_mac2.sh > /tmp/push_joint_waiter.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
RDJ=results/runs/can/joint_r34_bigunet
PAT="50_train_valloss.*joint_r34"

echo "[62] $(date '+%H:%M') attente fin du fine-tune prolongé (mac2 libre)..."
while true; do
  d=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 \
    "pgrep -f '50_train_valloss.*lookat_finetune_run31' >/dev/null && echo RUN || (grep -aoE '30000/30000' /tmp/run_lookat_ft_ext.log 2>/dev/null | tail -1)" 2>/dev/null)
  [ "$d" = "30000/30000" ] && break
  sleep 180
done
# garde-fou : plus aucun training mac2 en cours
while ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss' >/dev/null" 2>/dev/null; do sleep 60; done
echo "[62] $(date '+%H:%M') mac2 libre -> resume joint 50k->80k (LR constant)"

ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 EMA=1 EMA_DECAY=0.9999 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --config_path=$RDJ/checkpoints/last/pretrained_model/train_config.json \
    --resume=true --output_dir=$RDJ \
    --steps=80000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    >> /tmp/run_joint_ext.log 2>&1 & echo push_PID \$!"
sleep 12
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 32400 >/dev/null 2>&1 & echo \"[62] joint push lancé PID \$P + caffeinate 9h\"; \
    grep -ao '[0-9]*/80000' /tmp/run_joint_ext.log | tail -1; \
  else echo '[62] ⚠️ pas reparti :'; tail -15 /tmp/run_joint_ext.log; fi"
echo "[62] DONE (training lancé ; l'éval est gérée par 63)"
