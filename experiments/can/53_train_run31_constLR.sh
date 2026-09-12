#!/bin/bash
# run31-JUMEAU en LR CONSTANT (run31 était cosine) — même archi/données, pour lever le confond du cosine.
# Tourne sur mac2 dès que le fine-tune look-at finit (queue : joint+10k -> fine-tune -> run31-jumeau).
# En FICHIER (pas de self-match). nohup bash experiments/can/53_train_run31_constLR.sh > /tmp/run31cl_waiter.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
RUN=results/runs/can/run31_constLR
PAT="50_train_valloss.*run31_constLR"

echo "[53] $(date '+%H:%M') attente fin du fine-tune look-at sur mac2..."
while ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*lookat_finetune' >/dev/null" 2>/dev/null; do
  sleep 120
done
echo "[53] $(date '+%H:%M') fine-tune fini -> lancement run31-jumeau (LR constant, 40k) sur mac2"
EPS=$(python3 -c "print(list(range(150)))" | tr -d ' ')
ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.down_dims='[128,256,512]' --policy.vision_backbone='resnet34' \
    --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false \
    --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio_birdview \
    --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
    --dataset.episodes='$EPS' \
    --output_dir=$RUN \
    --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=2 > /tmp/run31_constLR.log 2>&1 & echo lance_PID \$!"
sleep 10
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 50400 >/dev/null 2>&1 & echo \"[53] run31-jumeau lancé PID \$P + caffeinate 14h\"; \
    grep -ao '[0-9]*/40000' /tmp/run31_constLR.log | tail -1; \
  else echo '[53] ⚠️ pas parti :'; tail -10 /tmp/run31_constLR.log; fi"
echo "[53] DONE"
