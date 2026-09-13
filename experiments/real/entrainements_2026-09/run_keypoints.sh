#!/usr/bin/env bash
# Balayage du NOMBRE DE KEYPOINTS du spatial softmax, sur le banc Can occlus 3 cm.
#
# Motivation : les keypoints du spatial softmax paraissent aléatoires (vérifié visuellement,
# et documenté : Soare montre qu ils ne sont pas sémantiquement interprétables). Mais leur
# NOMBRE est un levier mesuré fort chez lui : 8 -> 35,2 % | 32 -> 51,0 % | 128 -> 65,4 %.
# Notre config est à 32. Coût de passer à 128 : +61 536 params (+0,38 %) et +8 microsecondes.
#
# Un seul paramètre change entre les runs. Le reste est identique à la baseline (56,0 %).
set -u
cd ~/lerobot-experiments
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d " ")
for KP in 64 128 256; do
  OUT=results/runs/can/kp$KP
  echo "[$(date +%H:%M:%S)] ===== $KP keypoints ====="
  rm -rf $OUT
  $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion "--policy.down_dims=[64,128,256]" \
    --policy.spatial_softmax_num_keypoints=$KP \
    --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio_occ03 \
    --dataset.root=data_cache/lerobot_can_ph_proprio_occ03 --dataset.episodes="$EPS" \
    --output_dir=$OUT --batch_size=32 --steps=10000 --save_freq=5000 \
    --log_freq=500 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=4
  # cooldown, comme la baseline
  SRC=$OUT/checkpoints/010000/pretrained_model
  $PY -c "import json;c=json.load(open(\"$SRC/train_config.json\"));c.setdefault(\"wandb\",{})[\"enable\"]=False;c[\"wandb\"].pop(\"add_tags\",None);json.dump(c,open(\"/tmp/cd_kp$KP.json\",\"w\"))"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=/tmp/cd_kp$KP.json --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
  echo "[$(date +%H:%M:%S)] ===== $KP keypoints TERMINÉ ====="
done
echo "BALAYAGE TERMINÉ"
