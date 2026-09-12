#!/bin/bash
# REPRISE du delta chunk-wise depuis checkpoints/last (16000) -> 40000, puis cooldown 5k + eval.
# RELJOINT=1 (stats relatives cohérentes), CONST_LR 1e-4. Resume = optimiseur + step restaurés.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/joint_r34_reljoint_cw
CFG_RESUME=$RUN/checkpoints/last/pretrained_model/train_config.json

echo "[102b] $(date '+%H:%M') RESUME delta chunk-wise depuis $(readlink $RUN/checkpoints/last 2>/dev/null || echo last) -> 40000"
RELJOINT=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG_RESUME --resume=true \
  2>&1 | tee -a results/logs/can/run_reljoint_cw.log
echo "[102b] $(date '+%H:%M') TRAIN DONE (resume). Cooldown 40k + eval chunk-wise."

SRC=$RUN/checkpoints/040000/pretrained_model
[ -s "$SRC/model.safetensors" ] || { echo "[102b] ckpt 40k absent"; exit 1; }
CFG=/tmp/reljoint_cd.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
echo "[102b] $(date '+%H:%M') COOLDOWN 40k (LR 1e-4->0 / 5k)"
RELJOINT=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown_40k --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
$PY -u experiments/can/44_eval_reljoint.py --run-dir "$RUN/cooldown_40k" --steps 5000 --n 50 --kp 50 --out "$RUN/cooldown_40k/r.csv"
echo "[102b] $(date '+%H:%M') ===== VERDICT DELTA CHUNK-WISE 40k (cooldown) : $(tail -1 "$RUN/cooldown_40k/r.csv"|cut -d, -f4) (vs absolu 83%, séquentiel 0%) ====="
