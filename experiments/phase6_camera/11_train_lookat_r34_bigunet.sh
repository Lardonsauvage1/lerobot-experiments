#!/bin/bash
# Entraînement look-at — SUR MAC2 (le Principal est occupé par l'éval joint @500 ; 2 gros jobs MPS = OOM).
# Transfère le dataset look-at vers mac2, resync le code, lance l'entraînement (61M, jumeau run 31),
# avec un caffeinate LONG (15h) pour que mac2 ne se rendorme PAS avant qu'on récupère le checkpoint
# (leçon 24/06 : caffeinate lié au training -> mac2 dort à la fin -> impossible de transférer).
# Appelé par 12_lookat_pipeline.sh une fois la génération du dataset terminée.

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
IP=100.110.237.53
DS=data_cache/lerobot_can_ph_proprio_birdview_lookat
RUN=results/runs/phase6_camera/lookat_r34_bigunet
PAT="50_train_valloss.*lookat"

echo "[11] $(date '+%H:%M') transfert dataset look-at -> mac2..."
ok=0
for try in 1 2 3 4 5; do
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  if tar czf - "$DS" | ssh -o BatchMode=yes mac2 'cd ~/lerobot-experiments && tar xzf -'; then ok=1; break; fi
  echo "[11] retry transfert dataset ($try)..."; sleep 20
done
[ $ok -eq 1 ] || { echo "[11] ⚠️ transfert dataset ÉCHOUÉ — entraînement non lancé"; exit 1; }
# resync code (au cas où) — inclut le patch CONST_LR de 50_train_valloss
tar czf - experiments src requirements.txt | ssh -o BatchMode=yes mac2 'cd ~/lerobot-experiments && tar xzf -'

EPS=$(python3 -c "print(list(range(150)))" | tr -d ' ')
echo "[11] lancement entraînement sur mac2..."
ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.down_dims='[128,256,512]' --policy.vision_backbone='resnet34' \
    --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false \
    --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio_birdview_lookat \
    --dataset.root=data_cache/lerobot_can_ph_proprio_birdview_lookat \
    --dataset.episodes='$EPS' \
    --output_dir=$RUN \
    --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=2 > /tmp/run_lookat.log 2>&1 & echo lance_PID \$!"
sleep 8
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 54000 >/dev/null 2>&1 & echo \"[11] training mac2 PID \$P + caffeinate 15h\"; \
    grep -ao '[0-9]*/40000' /tmp/run_lookat.log | tail -1; \
  else echo '[11] ⚠️ training pas parti sur mac2 :'; tail -8 /tmp/run_lookat.log; fi"
echo "[11_lookat] LANCÉ SUR MAC2"
