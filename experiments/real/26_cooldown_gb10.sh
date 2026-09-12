#!/usr/bin/env bash
# Cooldown du ckpt final (30000) du run propre gb10 (dataset IMAGES) -> candidat déployable.
# LR 1e-4 -> 0 sur 5k (recette gagnante), EMA, GPU. Sortie apple_clean_gb10/cooldown_030000.
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}   # inoffensif (images)
RUN=results/runs/real/apple_clean_gb10
SRC=$RUN/checkpoints/030000/pretrained_model
[ -s "$SRC/model.safetensors" ] || SRC=$RUN/checkpoints/last/pretrained_model
OUT=$RUN/cooldown_030000
CFG=/tmp/apple_cd_gb10.json
[ -s "$SRC/model.safetensors" ] || { echo "ERREUR ckpt 30000 absent ($SRC)"; exit 1; }
python -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
echo "[cd-gb10 $(date '+%H:%M')] COOLDOWN ckpt 30000 : LR 1e-4 -> 0 / 5k, EMA, GPU -> $OUT"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 python -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=5000 --save_freq=1000 --log_freq=200 --eval_freq=0 2>&1 | tee /tmp/cd_gb10.log
echo "[cd-gb10 $(date '+%H:%M')] COOLDOWN DONE (rc=$?). brut=$OUT/checkpoints EMA=${OUT}_ema/checkpoints"
