#!/bin/bash
# PLAN DE SECOURS : mac2 est mort pendant run31-jumeau -> on le relocalise sur le Principal.
# Attend que le Principal soit libre (fin de l'éval fine-tune), puis :
#  - si mac2 est REVENU entre-temps : reprend run31-jumeau sur mac2 depuis son dernier checkpoint (plus rapide) ;
#  - sinon : entraîne run31-jumeau À NEUF sur le Principal (dataset clean local) puis rollouts descendants n=100.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP="${MAC2_HOST:-mac2}"
RUN=results/runs/can/run31_constLR
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

echo "[57] $(date '+%H:%M') attente que le Principal soit libre (fin éval fine-tune)..."
while pgrep -f "13_eval_camshift|32_eval_dense|56_eval_finetune" >/dev/null; do sleep 120; done
echo "[57] $(date '+%H:%M') Principal libre."

# mac2 revenu ? -> reprendre là-bas (plus rapide, et libère le Principal)
"$TS" ping -c 2 "$IP" >/dev/null 2>&1
if ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 'ls ~/lerobot-experiments/results/runs/can/run31_constLR/checkpoints/last >/dev/null 2>&1' 2>/dev/null; then
  echo "[57] mac2 REVENU -> reprise run31-jumeau sur mac2 depuis checkpoints/last"
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py --config_path=$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true --output_dir=$RUN --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 >> /tmp/run31_constLR.log 2>&1 & P=\$!; nohup caffeinate -t 50400 >/dev/null 2>&1 & echo repris_PID \$P"
  echo "[57] repris sur mac2 (le waiter 54 reprendra l'éval). DONE"; exit 0
fi

echo "[57] mac2 toujours mort -> run31-jumeau À NEUF sur le PRINCIPAL (LR constant, 40k)"
CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims='[128,256,512]' --policy.vision_backbone='resnet34' \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" --output_dir=$RUN \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=2 \
  > /tmp/run31cl_local_train.log 2>&1
echo "[57] entraînement Principal fini -> rollouts descendants n=100"
for s in 40000 38000 36000 34000 32000 30000 28000 26000 24000 22000 20000 18000 16000 14000 12000 10000 8000 6000 4000 2000; do
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir $RUN --steps "$s" --n 100 --out $RUN/rollouts_100.csv
done
echo "[57] DONE"
