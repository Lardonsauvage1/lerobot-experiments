#!/bin/bash
# FINE-TUNE look-at depuis run31 (warm-start) sur mac2, dès que le joint +10k finit.
# Hypothèse : partir d'un modèle qui SAIT déjà la tâche (run31, 94,8% clean) -> s'adapter aux
# variations de caméra est bien plus facile que d'apprendre tâche+robustesse de ZÉRO (échec à 2%).
# Mêmes I/O que run31 (proprio 9D, agentview+birdview, action 7D OSC) -> warm-start direct. LR constant.
# En FICHIER -> pas de bug pgrep self-match.
# Lancer : nohup bash experiments/phase6_camera/51_finetune_lookat_from_run31.sh > /tmp/ft_lookat_waiter.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
IP="${MAC2_HOST:-mac2}"
R31=results/runs/can/31_proprio_birdview_r34_bigunet/checkpoints/040000/pretrained_model
RUN=results/runs/phase6_camera/lookat_finetune_run31
PAT="50_train_valloss.*lookat_finetune_run31"

echo "[51] $(date '+%H:%M') attente fin du joint +10k sur mac2..."
while ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*joint_r34' >/dev/null" 2>/dev/null; do
  sleep 120
done
echo "[51] $(date '+%H:%M') joint +10k fini -> transfert run31 vers mac2"
ok=0
for try in 1 2 3 4 5; do
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  if tar czf - "$R31" | ssh -o BatchMode=yes mac2 'cd ~/lerobot-experiments && tar xzf -'; then ok=1; break; fi
  echo "[51] retry transfert run31 ($try)..."; sleep 20
done
[ $ok -eq 1 ] || { echo "[51] ⚠️ transfert run31 ÉCHOUÉ — fine-tune non lancé"; exit 1; }

EPS=$(python3 -c "print(list(range(150)))" | tr -d ' ')
echo "[51] lancement fine-tune look-at (warm-start run31, LR constant, 15k steps) sur mac2"
ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
  CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.down_dims='[128,256,512]' --policy.vision_backbone='resnet34' \
    --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false \
    --policy.crop_shape=null \
    --policy.pretrained_path=$R31 \
    --dataset.repo_id=local/can_ph_proprio_birdview_lookat \
    --dataset.root=data_cache/lerobot_can_ph_proprio_birdview_lookat \
    --dataset.episodes='$EPS' \
    --output_dir=$RUN \
    --batch_size=16 --steps=15000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=2 > /tmp/run_lookat_ft.log 2>&1 & echo lance_PID \$!"
sleep 10
ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
  if [ -n \"\$P\" ]; then nohup caffeinate -t 36000 >/dev/null 2>&1 & echo \"[51] fine-tune lancé PID \$P + caffeinate 10h\"; \
    grep -ao '[0-9]*/15000' /tmp/run_lookat_ft.log | tail -1; \
  else echo '[51] ⚠️ pas parti :'; tail -10 /tmp/run_lookat_ft.log; fi"
echo "[51] DONE"
