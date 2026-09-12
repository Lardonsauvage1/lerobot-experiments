#!/bin/bash
# Cooldown du MEILLEUR checkpoint (2000, min val-loss=0.0214) -> modele deployable.
# LR 1e-4 -> 0 lineaire sur 5k steps, EMA active. Sur Principal (MPS). Ckpt 2000 = branche commune.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/real/apple_joint_224_r34
SRC=$RUN/checkpoints/002000/pretrained_model
OUT=$RUN/cooldown_002000
CDLOG=results/logs/real/cooldown.log
mkdir -p results/logs/real

log(){ echo "[cooldown $(date '+%m-%d %H:%M')] $*" | tee -a "$CDLOG"; }

[ -s "$SRC/model.safetensors" ] || { log "ERREUR ckpt 2000 absent"; exit 1; }
CFG=/tmp/apple_cd_2000.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"

log "COOLDOWN ckpt 2000 : LR 1e-4 -> 0 / 5k, EMA, MPS -> $OUT"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=5000 --save_freq=1000 --log_freq=200 --eval_freq=0 >> "$CDLOG" 2>&1
rc=$?
log "COOLDOWN DONE (rc=$rc). Modele final : brut=$OUT/checkpoints, EMA=${OUT}_ema/checkpoints"
log "Deployable = min val-loss du cooldown (voir 'val_loss:' dans $CDLOG) ; comparer aussi a l'EMA@2000 pre-cooldown."
