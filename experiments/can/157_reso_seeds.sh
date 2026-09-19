#!/usr/bin/env bash
# TEST DE RÉSOLUTION PROPRE — tout figé sauf la résolution.
#
# Pourquoi ce script existe : la comparaison 84px (96,0 %) vs 224px (72,4 %) de la phase
# precedente n'etait PAS un test de resolution. Les deux bras differaient aussi par le
# backbone (R18 vs R34), le U-Net ([512,1024,2048] vs [128,256,512]), le batch (64 vs 16),
# les pas (40k vs 30k) et le nombre de demos (196 vs 150) : 5,3x d'ecart de budget.
#
# Ici : meme backbone, meme U-Net, meme batch, memes pas, memes demos, meme LR, meme EMA,
# meme cooldown, meme RATIO de recadrage (0,905). SEULE la resolution change.
# Ordre : une graine complete sur les 3 resolutions avant de passer a la suivante, pour
# qu'une interruption laisse toujours un balayage exploitable.
set -u
cd ~/lerobot-experiments || exit 1
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
for SEED in 42 43 44; do
for RES in 84 96 224; do
  case $RES in
    84)  DS=lerobot_can_ph_proprio_84_img;  ID=local/can_ph_proprio_84_img;  CROP=76  ;;
    96)  DS=lerobot_can_ph_proprio_img;     ID=local/can_ph_proprio_img;     CROP=87  ;;
    224) DS=lerobot_can_ph_proprio_224_img; ID=local/can_ph_proprio_224_img; CROP=203 ;;
  esac
  RUN=results/runs/can/reso_${RES}_s${SEED}
  [ -f $RUN/cooldown/checkpoints/005000/pretrained_model/model.safetensors ] && { echo "[reso $RES s$SEED] deja fait, skip"; continue; }
  echo "[reso $RES s$SEED $(date '+%H:%M')] crop $CROP<-$RES, R18, U-Net[64,128,256], batch 32, 10k"
  EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.vision_backbone=resnet18 --policy.down_dims="[64,128,256]" \
    --policy.crop_shape="[$CROP,$CROP]" --policy.crop_is_random=true \
    --policy.device=xpu --policy.push_to_hub=false \
    --dataset.repo_id=$ID --dataset.root=data_cache/$DS --dataset.episodes="$EPS" \
    --output_dir=$RUN \
    --batch_size=32 --steps=10000 --save_freq=10000 --log_freq=500 --eval_freq=0 \
    --seed=$SEED --wandb.enable=false --num_workers=8 || { echo "[reso $RES s$SEED] ECHEC TRAIN"; continue; }
  SRC=$RUN/checkpoints/010000/pretrained_model
  CFG=/tmp/cd_reso_${RES}_${SEED}.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
  EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$RUN/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0 \
    || echo "[reso $RES s$SEED] ECHEC COOLDOWN"
  echo "[reso $RES s$SEED $(date '+%H:%M')] DONE"
done
done
echo "RESO TRAIN DONE"
