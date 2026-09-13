#!/usr/bin/env bash
# Teste KPAMP : donner au réseau l AMPLITUDE d activation de chaque keypoint, en plus de
# ses coordonnées. Le spatial softmax normalise, donc il ne peut jamais dire « je ne vois
# pas l objet » — quand la canette disparaît, les keypoints se replacent silencieusement
# (mesuré : 3,3 px en moyenne, 42,8 px au pire). L amplitude est l information jetée.
# Tout le reste est identique à la baseline (56,0 %). 32 keypoints, comme elle.
set -u
cd ~/lerobot-experiments
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d " ")
OUT=results/runs/can/kpamp
# attendre la fin du balayage keypoints (une seule GPU)
while pgrep -f run_keypoints.sh > /dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] ===== KPAMP (32 kp + amplitude) ====="
rm -rf $OUT
KPAMP=1 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion "--policy.down_dims=[64,128,256]" \
  --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_occ03 \
  --dataset.root=data_cache/lerobot_can_ph_proprio_occ03 --dataset.episodes="$EPS" \
  --output_dir=$OUT --batch_size=32 --steps=10000 --save_freq=5000 \
  --log_freq=500 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=4
SRC=$OUT/checkpoints/010000/pretrained_model
$PY -c "import json;c=json.load(open(\"$SRC/train_config.json\"));c.setdefault(\"wandb\",{})[\"enable\"]=False;c[\"wandb\"].pop(\"add_tags\",None);json.dump(c,open(\"/tmp/cd_kpamp.json\",\"w\"))"
KPAMP=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=/tmp/cd_kpamp.json --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
echo "[$(date +%H:%M:%S)] ===== KPAMP TERMINÉ ====="
