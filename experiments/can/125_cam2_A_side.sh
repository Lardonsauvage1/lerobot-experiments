#!/usr/bin/env bash
# CAM2 bras A — caméra de CÔTÉ seule (agentview). Config du run 02 (prouvée 70,4 % @500)
# + EMA et cooldown (règles permanentes). Jumeau exact de 126 sauf la 2e caméra.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
RUN=results/runs/can/cam2_A_side
echo "[cam2-A $(date '+%H:%M')] côté seule, R18, [64,128,256], batch 32, 10k, EMA"
EMA=1 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims="[64,128,256]" \
  --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio --dataset.root=data_cache/lerobot_can_ph_proprio \
  --dataset.episodes="$EPS" --output_dir=$RUN \
  --batch_size=32 --steps=10000 --save_freq=2000 --log_freq=200 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[cam2-A $(date '+%H:%M')] TRAIN DONE -> cooldown"
SRC=$RUN/checkpoints/010000/pretrained_model
CFG=/tmp/cd_cam2A.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown --steps=5000 --save_freq=5000 --log_freq=200 --eval_freq=0
echo "[cam2-A $(date '+%H:%M')] COOLDOWN DONE"
