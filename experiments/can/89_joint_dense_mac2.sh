#!/bin/bash
# mac2 (densification joint) : entraîne les cooldowns COOLD_MAC2={2,4,12,28} (5k, LR 1e-4->0, save LR=0).
# Garde localement ; Principal (88) tire et évalue. En FICHIER.
# Lancer (mac2) : caffeinate -i nohup bash experiments/can/89_joint_dense_mac2.sh > /tmp/dense_mac2.log 2>&1 &
set -u
cd ~/lerobot-experiments 2>/dev/null || cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
PY=venv312/bin/python
SRC_RUN=results/runs/can/joint_r34_bigunet
MAC2="2 4 12 16 24 28"

# garde-fou disque : un cooldown joint écrit ~1,4 Go ; on attend s'il reste < 8 Go (au lieu de planter)
wait_for_disk() {
  while [ "$(df -m . | awk 'NR==2{print $4}')" -lt 8000 ]; do
    echo "[89] $(date '+%H:%M') ⏸ disque < 8 Go libre -> pause 5 min (libère de l'espace)"; sleep 300
  done
}

for K in $MAC2; do
  wait_for_disk
  OUT=results/runs/can/joint_cd_${K}k
  [ -s "$OUT/checkpoints/005000/pretrained_model/model.safetensors" ] && { echo "[89] ${K}k déjà fait"; continue; }
  S=$(printf '%06d' $((K*1000))); SRC=$SRC_RUN/checkpoints/$S/pretrained_model
  [ -s "$SRC/model.safetensors" ] || { echo "[89] source ${K}k absente"; continue; }
  CFG=/tmp/cd_config_${K}.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
  echo "[89] $(date '+%H:%M') COOLDOWN depuis ${K}k"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
  echo "[89] $(date '+%H:%M') ${K}k prêt"
done
echo "[89] $(date '+%H:%M') DONE"
