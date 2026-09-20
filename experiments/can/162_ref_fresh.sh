#!/usr/bin/env bash
# RÉFÉRENCE HOMOGÈNE : [64,128,256] avec encodeur poignet NEUF, graines 43 et 44.
# Pourquoi : le groupe de référence actuel (I/I2/I3) part d'un poignet REPRIS, alors que
# le balayage de décodeur part d'un poignet NEUF. La comparaison mélangeait donc deux
# changements. Seul J (graine 42) est apparié en variante — il manque 43 et 44.
# Attend la fin du balayage avant de démarrer.
set -u
cd ~/lerobot-experiments || exit 1
PY=~/ipex_test_venv/bin/python
for i in $(seq 1 600); do
  grep -q "UNET SWEEP DONE" results/logs/can/balayage_unet.log 2>/dev/null && break
  sleep 60
done
grep -q "UNET SWEEP DONE" results/logs/can/balayage_unet.log 2>/dev/null || { echo "TIMEOUT balayage"; exit 1; }
echo "[ref-fresh $(date '+%H:%M')] balayage fini -> reference poignet NEUF"
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
for SEED in 43 44; do
  RUN=results/runs/can/ref_fresh_s$SEED
  [ -f $RUN/cooldown/checkpoints/005000/pretrained_model/model.safetensors ] && { echo "[s$SEED] deja fait"; continue; }
  echo "[ref-fresh s$SEED $(date '+%H:%M')] [64,128,256], poignet NEUF, batch 32"
  EMA=1 FREEZE_CAM0=1 $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.down_dims="[64,128,256]" \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.pretrained_path=results/runs/can/cam2_I_frozen_init_fresh/pretrained_model \
    --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio_wrist_fixstats \
    --dataset.root=data_cache/lerobot_can_ph_proprio_wrist_fixstats \
    --dataset.episodes="$EPS" --output_dir=$RUN \
    --batch_size=32 --steps=10000 --save_freq=10000 --log_freq=500 --eval_freq=0 \
    --seed=$SEED --wandb.enable=false --num_workers=8 || { echo "[s$SEED] ECHEC"; continue; }
  SRC=$RUN/checkpoints/010000/pretrained_model
  CFG=/tmp/cd_ref_$SEED.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
  EMA=1 FREEZE_CAM0=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$RUN/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
  echo "[ref-fresh s$SEED $(date '+%H:%M')] DONE"
done
echo "REF FRESH DONE"
