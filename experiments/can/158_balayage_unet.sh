#!/usr/bin/env bash
# BALAYAGE DE TAILLE DE U-NET, sur la VRAIE configuration cible :
#   2 cameras = vue exterieure (que le bras masque par moments) + camera EMBARQUEE au poignet,
#   protocole a ENCODEUR GELE (sans lui, ajouter le poignet coute 25 points).
# Seul `down_dims` varie. Tout le reste est fige : ResNet18, 96 px, 150 demos, batch 32,
# 10k pas + recuit 5k, LR constante 1e-4, EMA.
#
# Reference deja mesuree : [64,128,256] = 86,2 % (4 graines : 86,2 / 91,2 / 83,0 / 84,2).
# On ajoute donc seulement les deux tailles superieures.
#
# Chaque (taille, graine) = 4 etapes :
#   1. mono vue exterieure          (10k + recuit 5k)
#   2. moule bi-camera aux bonnes formes (1 pas, jetable)
#   3. conversion mono -> bi-camera, colonnes poignet a zero, encodeur poignet NEUF
#   4. bi-camera a encodeur gele    (10k + recuit 5k)
set -u
cd ~/lerobot-experiments || exit 1
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
COMMUN="--policy.type=diffusion --policy.vision_backbone=resnet18 \
  --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.episodes=$EPS --batch_size=32 --log_freq=500 --eval_freq=0 \
  --wandb.enable=false --num_workers=8"
MONO_DS="--dataset.repo_id=local/can_ph_proprio_fixstats --dataset.root=data_cache/lerobot_can_ph_proprio_fixstats"
BI_DS="--dataset.repo_id=local/can_ph_proprio_wrist_fixstats --dataset.root=data_cache/lerobot_can_ph_proprio_wrist_fixstats"

recuit () {  # $1 = run, $2 = extra env
  SRC=$1/checkpoints/010000/pretrained_model
  CFG=/tmp/cd_$(echo $1 | tr '/' '_').json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
  env $2 EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$1/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
}

for SEED in 42 43; do
for DD in "[128,256,512]" "[256,512,1024]"; do
  TAG=$(echo $DD | tr -d '[],' | tr ' ' '_')
  M=results/runs/can/unet_${TAG}_mono_s${SEED}
  B=results/runs/can/unet_${TAG}_gele_s${SEED}
  [ -f $B/cooldown/checkpoints/005000/pretrained_model/model.safetensors ] && { echo "[$TAG s$SEED] deja fait"; continue; }

  echo "[$TAG s$SEED $(date '+%H:%M')] 1/4 mono vue exterieure"
  if [ ! -f $M/cooldown/checkpoints/005000/pretrained_model/model.safetensors ]; then
    EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py $COMMUN $MONO_DS \
      --policy.down_dims="$DD" --output_dir=$M --steps=10000 --save_freq=10000 --seed=$SEED || { echo "[$TAG s$SEED] ECHEC mono"; continue; }
    recuit $M "EMA=1"
  fi

  echo "[$TAG s$SEED $(date '+%H:%M')] 2/4 moule bi-camera"
  MOULE=results/runs/can/unet_${TAG}_moule
  if [ ! -f $MOULE/checkpoints/000001/pretrained_model/model.safetensors ]; then
    $PY -u experiments/lift/50_train_valloss.py $COMMUN $BI_DS \
      --policy.down_dims="$DD" --policy.use_separate_rgb_encoder_per_camera=true \
      --output_dir=$MOULE --steps=1 --save_freq=1 --seed=0 || { echo "[$TAG] ECHEC moule"; continue; }
  fi

  echo "[$TAG s$SEED $(date '+%H:%M')] 3/4 conversion mono -> bi-camera"
  INIT=results/runs/can/unet_${TAG}_init_s${SEED}/pretrained_model
  FRESH_WRIST=1 INIT_MONO=$M/cooldown/checkpoints/005000/pretrained_model \
    INIT_MOULE=$MOULE/checkpoints/000001/pretrained_model INIT_OUT=$INIT \
    $PY -u experiments/can/141_init_from_mono.py || { echo "[$TAG s$SEED] ECHEC conversion"; continue; }

  echo "[$TAG s$SEED $(date '+%H:%M')] 4/4 bi-camera a encodeur GELE"
  EMA=1 FREEZE_CAM0=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py $COMMUN $BI_DS \
    --policy.down_dims="$DD" --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.pretrained_path=$INIT --output_dir=$B --steps=10000 --save_freq=10000 --seed=$SEED \
    || { echo "[$TAG s$SEED] ECHEC gele"; continue; }
  recuit $B "EMA=1 FREEZE_CAM0=1"
  echo "[$TAG s$SEED $(date '+%H:%M')] DONE"
done
done
echo "UNET SWEEP DONE"
