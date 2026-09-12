#!/bin/bash
# mac2 (parallélisation du sweep cooldown) : entraîne les cooldowns MAC2={20,40,60}
# (5k, LR 1e-4->0, save_freq=5000 -> seul le checkpoint LR=0). Garde localement ; Principal (84) le tire et évalue.
# Attend la fin de B (pré-entraîné). En FICHIER.
# Lancer (mac2) : caffeinate -i nohup bash experiments/can/86_cooldown_mac2.sh > /tmp/cd_mac2.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
SRC_RUN=results/runs/can/joint_r34_bigunet
MAC2="80"

echo "[86] $(date '+%H:%M') attente fin de B (pré-entraîné)..."
while pgrep -f "81_train_joint_pretrained|50_train_valloss.*pretrained" >/dev/null; do sleep 120; done
echo "[86] $(date '+%H:%M') B fini -> cooldowns mac2"

for K in $MAC2; do
  OUT=results/runs/can/joint_cd_${K}k
  [ -s "$OUT/checkpoints/005000/pretrained_model/model.safetensors" ] && { echo "[86] ${K}k déjà fait"; continue; }
  S=$(printf '%06d' $((K*1000))); SRC=$SRC_RUN/checkpoints/$S/pretrained_model
  [ -s "$SRC/model.safetensors" ] || { echo "[86] source ${K}k absente"; continue; }
  CFG=/tmp/cd_config_${K}.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
  echo "[86] $(date '+%H:%M') COOLDOWN depuis ${K}k (LR 1e-4->0 sur 5k)"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
  echo "[86] $(date '+%H:%M') ${K}k prêt (Principal tirera le checkpoint LR=0)"
done
echo "[86] $(date '+%H:%M') DONE"
