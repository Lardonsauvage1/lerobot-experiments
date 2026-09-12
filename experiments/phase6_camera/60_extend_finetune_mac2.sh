#!/bin/bash
# PROLONGATION du fine-tune look-at : resume 15k -> 30k (LR constant 1e-4), sur mac2,
# DÈS que run31-jumeau libère mac2. But : trancher si plus de steps remontent vers run31
# (le graphe frise suggère que non). En FICHIER -> pas de bug pgrep self-match.
# Lancer : nohup bash experiments/phase6_camera/60_extend_finetune_mac2.sh > /tmp/ext_ft_waiter.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
RUN=results/runs/phase6_camera/lookat_finetune_run31
PAT="50_train_valloss.*lookat_finetune_run31"

echo "[60] $(date '+%H:%M') attente fin run31-jumeau (mac2 libre)..."
while ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*run31_constLR' >/dev/null" 2>/dev/null; do
  sleep 120
done
echo "[60] $(date '+%H:%M') mac2 libre -> resume fine-tune look-at 15k->30k (LR constant)"

ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --config_path=$RUN/checkpoints/last/pretrained_model/train_config.json \
    --resume=true --output_dir=$RUN \
    --steps=30000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    >> /tmp/run_lookat_ft_ext.log 2>&1 & echo resume_PID \$!"
sleep 12
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 21600 >/dev/null 2>&1 & echo \"[60] fine-tune ext lancé PID \$P + caffeinate 6h\"; \
    grep -ao '[0-9]*/30000' /tmp/run_lookat_ft_ext.log | tail -1; \
  else echo '[60] ⚠️ pas reparti :'; tail -15 /tmp/run_lookat_ft_ext.log; fi"
echo "[60] DONE (training lancé ; l'éval est gérée par 61)"
