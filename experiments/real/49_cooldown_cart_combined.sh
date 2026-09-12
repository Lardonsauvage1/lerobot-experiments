#!/usr/bin/env bash
# Cooldown apple_cart_combined_128 (LR 1e-4->0/5k, EMA). Config repris du checkpoint.
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
RUN=results/runs/real/apple_cart_combined_128
STEPS=${STEPS:-40000}
SRC=$RUN/checkpoints/$(printf '%06d' $STEPS)/pretrained_model
[ -s "$SRC/model.safetensors" ] || SRC=$RUN/checkpoints/last/pretrained_model
CFG=/tmp/cd_cart_combined.json
python -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
echo "[cd-cart-comb $(date '+%H:%M')] cooldown 5k depuis $SRC"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 python -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown --steps=5000 --save_freq=1000 --log_freq=50 --eval_freq=0 2>&1 | tee /tmp/cd_cart_combined.log
echo "[cd-cart-comb $(date '+%H:%M')] COOLDOWN DONE"
