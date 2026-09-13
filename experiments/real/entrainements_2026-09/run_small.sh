#!/usr/bin/env bash
# PLUS PETIT MODÈLE + PLUS DE KEYPOINTS : peut-on rendre dans le U-Net ce qu on gagne
# dans le spatial softmax ? L encodeur pèse 11,2 M (70 % du modèle) et ne bouge presque
# pas avec le nombre de keypoints ; c est le U-Net qui est compressible.
#   [64,128,256]+32kp (baseline) = 16,14 M -> 56,0 %
#   [32,64,128]+128kp            = 13,58 M (-16 %) -> ?
set -u
cd ~/lerobot-experiments
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d " ")
while pgrep -f "run_keypoints.sh|run_kpamp.sh|run_oracle.sh" > /dev/null; do sleep 120; done
for CFG in "64 small64" "128 small128"; do
  set -- $CFG; KP=$1; TAG=$2
  OUT=results/runs/can/$TAG
  echo "[$(date +%H:%M:%S)] ===== petit U-Net [32,64,128] + $KP keypoints ====="
  rm -rf $OUT
  $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion "--policy.down_dims=[32,64,128]" \
    --policy.spatial_softmax_num_keypoints=$KP \
    --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio_occ03 \
    --dataset.root=data_cache/lerobot_can_ph_proprio_occ03 --dataset.episodes="$EPS" \
    --output_dir=$OUT --batch_size=32 --steps=10000 --save_freq=5000 \
    --log_freq=500 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=4
  SRC=$OUT/checkpoints/010000/pretrained_model
  $PY -c "import json;c=json.load(open(\"$SRC/train_config.json\"));c.setdefault(\"wandb\",{})[\"enable\"]=False;c[\"wandb\"].pop(\"add_tags\",None);json.dump(c,open(\"/tmp/cd_$TAG.json\",\"w\"))"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=/tmp/cd_$TAG.json --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
  echo "[$(date +%H:%M:%S)] ===== $TAG TERMINÉ ====="
done
