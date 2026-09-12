#!/bin/bash
# Cooldown COURT (1000 steps) du ckpt 2000 : anneal LR 1e-4->0 raide -> candidat "moins de derive overfit".
# A comparer au cooldown 5k et a l'EMA@2000. Sur Principal (MPS), enchaine APRES le cooldown 5k.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/real/apple_joint_224_r34
SRC=$RUN/checkpoints/002000/pretrained_model
OUT=$RUN/cooldown_002000_short1k
CDLOG=results/logs/real/cooldown_short1k.log
mkdir -p results/logs/real
log(){ echo "[cd-short $(date '+%m-%d %H:%M')] $*" | tee -a "$CDLOG"; }

[ -s "$SRC/model.safetensors" ] || { log "ERREUR ckpt 2000 absent"; exit 1; }
CFG=/tmp/apple_cd_2000_short.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"

log "COOLDOWN COURT ckpt 2000 : LR 1e-4 -> 0 / 1000, EMA, MPS -> $OUT"
EMA=1 COOLDOWN_STEPS=1000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=1000 --save_freq=500 --log_freq=100 --eval_freq=0 >> "$CDLOG" 2>&1
log "COOLDOWN COURT DONE (rc=$?). Candidat: brut=$OUT/checkpoints, EMA=${OUT}_ema/checkpoints"
