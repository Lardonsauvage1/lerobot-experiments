#!/bin/bash
# COOLDOWN de B (joint pré-entraîné) depuis son meilleur checkpoint (24k, brut 18%).
# LR par groupe (encodeur 1e-5 / U-Net 1e-4) -> 0 sur 5k steps (cooldown préservant le ratio).
# Rapatrie le ckpt 24k depuis mac2, cooldown sur Principal, éval. Compare : from-scratch cooldown = 83%.
# Lancer : nohup caffeinate -i bash experiments/can/99_cooldown_B.sh > /tmp/cooldown_B.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
SRC=results/runs/can/joint_r34_pretrained/checkpoints/024000/pretrained_model
OUT=results/runs/can/joint_pretrained_cooldown_24k

echo "[99] $(date '+%H:%M') rapatriement ckpt 24k de B depuis mac2"
mkdir -p "$(dirname $SRC)"
[ -s "$SRC/model.safetensors" ] || ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $SRC" | tar xzf - -C . 2>/dev/null
[ -s "$SRC/model.safetensors" ] || { echo "[99] ckpt 24k absent, abandon"; exit 1; }

CFG=/tmp/cd_B_config.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"

echo "[99] $(date '+%H:%M') COOLDOWN par groupe (encodeur 1e-5 / U-Net 1e-4 -> 0 sur 5k) depuis B@24k"
ENCODER_LR=1e-5 UNET_LR=1e-4 COOLDOWN_STEPS=5000 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0

echo "[99] $(date '+%H:%M') éval du cooldown B (LR=0, n=50)"
if [ -s "$OUT/checkpoints/005000/pretrained_model/model.safetensors" ]; then
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$OUT" --steps 5000 --n 50 --kp 50 --out "$OUT/r.csv"
  echo "[99] $(date '+%H:%M') ===== COOLDOWN B : $(tail -1 "$OUT/r.csv"|cut -d, -f4) (vs brut 18% ; from-scratch cooldown 83%) ====="
fi
rm -rf "$OUT/checkpoints" results/runs/can/joint_r34_pretrained/checkpoints/024000