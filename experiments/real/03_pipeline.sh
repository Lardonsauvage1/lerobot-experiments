#!/bin/bash
# Orchestrateur AUTONOME tache pomme (donnees reelles) :
#   1) attend la fin de la conversion mcap->LeRobot (02_convert_all.py)
#   2) verifie le dataset (nb episodes, info.json)
#   3) lance l'entrainement 40k (joint absolu 6D, R34 + gros U-Net, 224px, batch16, LR const 1e-4, EMA)
#   4) enchaine un cooldown 5k (LR 1e-4 -> 0) sur le checkpoint 40k
# Pas de rollout (pas de simulateur) : le juge offline = val-loss (episodes 120..137, auto hors --episodes).
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/real/apple_joint_224_r34
CONVLOG=results/logs/real/convert.log
TLOG=results/logs/real/train.log
DSROOT=data_cache/lerobot_apple_joint_224
mkdir -p results/logs/real

log(){ echo "[pipeline $(date '+%m-%d %H:%M')] $*" | tee -a "$TLOG"; }

# --- 1) attendre la conversion ---
log "attente fin conversion..."
while pgrep -f 02_convert_all.py >/dev/null; do sleep 30; done
if ! grep -q "TERMINE" "$CONVLOG"; then
  log "ERREUR : conversion terminee SANS 'TERMINE' (crash ?). Voir $CONVLOG. STOP."; exit 1
fi

# --- 2) verifier le dataset ---
NEP=$($PY -c "import json;print(json.load(open('$DSROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
log "dataset construit : $NEP episodes"
if [ "$NEP" -lt 40 ]; then log "ERREUR : seulement $NEP episodes (<40). STOP."; exit 1; fi

# --- 3) entrainement 30k (budget cale sur ~12k frames / 46 ep : train 40 / val 6) ---
NTRAIN=40
# nettoie un RUN vide (sans checkpoints) laisse par une tentative avortee -> evite FileExistsError
[ -d "$RUN" ] && [ ! -d "$RUN/checkpoints" ] && { log "RUN vide detecte -> nettoyage $RUN"; rm -rf "$RUN"; }
EPS=$($PY -c "print(list(range($NTRAIN)))" | tr -d ' ')   # train 0..39 ; val = 40..NEP-1
log "TRAIN 30k : R34 + U-Net[128,256,512], 2 cams 224px, batch16, LR const 1e-4, EMA, train=${NTRAIN}ep val=$((NEP-NTRAIN))ep"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims="[128,256,512]" --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/apple_joint_224 --dataset.root=$DSROOT --dataset.episodes="$EPS" \
  --output_dir=$RUN --batch_size=16 --steps=30000 --save_freq=2000 --log_freq=500 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 >> "$TLOG" 2>&1
log "TRAIN 30k DONE (rc=$?)"

# --- 4) cooldown 5k sur le ckpt 30k ---
S=030000
SRC=$RUN/checkpoints/$S/pretrained_model
if [ ! -s "$SRC/model.safetensors" ]; then log "ERREUR : ckpt $S absent, pas de cooldown. STOP."; exit 1; fi
CFG=/tmp/apple_cd_$S.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
log "COOLDOWN $S : LR 1e-4 -> 0 sur 5k (EMA aussi)"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown_$S --steps=5000 --save_freq=5000 --log_freq=200 --eval_freq=0 >> "$TLOG" 2>&1
log "COOLDOWN DONE (rc=$?). Checkpoints: brut=$RUN/checkpoints, EMA=${RUN}_ema/checkpoints, cooldown=$RUN/cooldown_$S"
log "PIPELINE TERMINE. Juge final = deploiement reel + val-loss (voir 'val_loss:' dans $TLOG)."
