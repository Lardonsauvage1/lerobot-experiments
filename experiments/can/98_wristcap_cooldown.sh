#!/usr/bin/env bash
# WRISTCAP — cooldown générique (LR 1e-4 -> 0 sur 5k, EMA) du checkpoint final d'un bras.
# OBLIGATOIRE avant éval (feedback : jamais évaluer le brut seul). Sur gb10.
# Usage : bash 98_wristcap_cooldown.sh wristcap_A_agentview [SRCSTEP=030000]
set -u
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
RUN=$1
SRCSTEP=${2:-030000}
SRC=results/runs/can/$RUN/checkpoints/$SRCSTEP/pretrained_model
[ -s "$SRC/model.safetensors" ] || SRC=results/runs/can/$RUN/checkpoints/last/pretrained_model
[ -s "$SRC/model.safetensors" ] || { echo "ERREUR ckpt absent pour $RUN"; exit 1; }
OUT=results/runs/can/$RUN/cooldown
CFG=/tmp/cd_$RUN.json
python -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
echo "[cd $RUN $(date '+%H:%M')] LR 1e-4 -> 0 / 5k, EMA -> $OUT"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 python -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=5000 --save_freq=1000 --log_freq=200 --eval_freq=0 2>&1 | tee /tmp/cd_$RUN.log
echo "[cd $RUN $(date '+%H:%M')] COOLDOWN DONE — candidat éval : $OUT/checkpoints/005000/pretrained_model"
