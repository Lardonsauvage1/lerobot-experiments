#!/usr/bin/env bash
# ORACLE-MÉMOIRE — la BORNE SUPÉRIEURE qu on n a jamais mesurée.
#
# On donne à la policy la position VRAIE de la canette en permanence, même quand elle est
# masquée à l image (state 12D au lieu de 9D). Ce n est pas déployable : c est un plafond.
#
#   +30 points vs baseline -> l information manquante vaut cher, et notre module mémoire
#                             était simplement mauvais. CAMP mérite d être refait.
#   +2 points              -> aucune mémoire n aidera sur ce banc, et l absence d effet de
#                             CAMP n a rien d étonnant. Le chantier mémoire s arrête ici.
#
# Config identique à la baseline (56,0 %) : ResNet18 + [64,128,256], cosine, batch 32, 10k.
set -u
cd ~/lerobot-experiments
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d " ")
OUT=results/runs/can/oracle12d
while pgrep -f "run_keypoints.sh|run_kpamp.sh" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] ===== ORACLE 12D ====="
rm -rf $OUT
$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion "--policy.down_dims=[64,128,256]" \
  --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_occ03 \
  --dataset.root=data_cache/lerobot_can_ph_occ03 --dataset.episodes="$EPS" \
  --output_dir=$OUT --batch_size=32 --steps=10000 --save_freq=5000 \
  --log_freq=500 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=4
SRC=$OUT/checkpoints/010000/pretrained_model
$PY -c "import json;c=json.load(open(\"$SRC/train_config.json\"));c.setdefault(\"wandb\",{})[\"enable\"]=False;c[\"wandb\"].pop(\"add_tags\",None);json.dump(c,open(\"/tmp/cd_oracle.json\",\"w\"))"
COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=/tmp/cd_oracle.json --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
echo "[$(date +%H:%M:%S)] ===== ORACLE TERMINÉ ====="
