#!/bin/bash
# Cooldown du checkpoint 12000 (run prolonge, LR const river-valley) -> candidat modele deployable.
# LR 1e-4 -> 0 sur 5k (meme recette que le cooldown 5k gagnant depuis 2000), EMA, Principal MPS.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/real/apple_joint_224_r34
SRC=$RUN/checkpoints/012000/pretrained_model
OUT=$RUN/cooldown_012000
CDLOG=results/logs/real/cooldown_12000.log
mkdir -p results/logs/real
log(){ echo "[cd12k $(date '+%m-%d %H:%M')] $*" | tee -a "$CDLOG"; }

[ -s "$SRC/model.safetensors" ] || { log "ERREUR ckpt 12000 absent"; exit 1; }
CFG=/tmp/apple_cd_12000.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"

log "COOLDOWN ckpt 12000 : LR 1e-4 -> 0 / 5k, EMA, MPS -> $OUT"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=5000 --save_freq=1000 --log_freq=200 --eval_freq=0 >> "$CDLOG" 2>&1
log "COOLDOWN 12k DONE (rc=$?). Candidat : brut=$OUT/checkpoints, EMA=${OUT}_ema/checkpoints"
