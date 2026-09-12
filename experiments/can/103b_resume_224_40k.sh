#!/bin/bash
# REPRISE 224 delta chunk-wise : 20k -> 40k (batch 16, cohérent avec le 42%@20k), puis cooldown 40k + eval 224.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/joint_r34_reljoint_cw_224
RELSTATS=data_cache/lerobot_can_ph_joint_birdview/meta/relstats_chunkwise.json
CFG_RESUME=$RUN/checkpoints/last/pretrained_model/train_config.json

echo "[103b] $(date '+%H:%M') RESUME 224 depuis $(readlink $RUN/checkpoints/last 2>/dev/null) -> 40000 (batch 16)"
RELJOINT=1 RELJOINT_STATS=$RELSTATS CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG_RESUME --resume=true --steps=40000 \
  2>&1 | tee -a results/logs/can/run_reljoint_cw_224.log
echo "[103b] $(date '+%H:%M') TRAIN DONE (40k). Cooldown + eval."

S=040000
SRC=$RUN/checkpoints/$S/pretrained_model
[ -s "$SRC/model.safetensors" ] || { echo "[103b] ckpt $S absent"; exit 1; }
CFG=/tmp/reljoint224_cd_$S.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
echo "[103b] $(date '+%H:%M') COOLDOWN $S (LR 1e-4->0 / 5k)"
RELJOINT=1 RELJOINT_STATS=$RELSTATS COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown_$S --steps=5000 --save_freq=5000 --log_freq=200 --eval_freq=0
$PY -u experiments/can/44_eval_reljoint.py --run-dir "$RUN/cooldown_$S" --steps 5000 --n 50 --kp 50 --img-size 224 --out "$RUN/cooldown_$S/r.csv"
echo "[103b] $(date '+%H:%M') ===== VERDICT 224 delta chunk-wise 40k (cooldown) : $(tail -1 "$RUN/cooldown_$S/r.csv"|cut -d, -f4) (vs 20k=42%, 96px=6%, absolu=92%) ====="
