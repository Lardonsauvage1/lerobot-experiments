#!/bin/bash
# Cooldown court (1000 steps) sur ATOMMAN en CPU (Intel Ultra 9, pas de CUDA).
# Threads optimises : OMP=12 + taskset P-cores 0-11 (config optimale connue). device=cpu.
# A executer SUR atomman : cd ~/lerobot-experiments && bash experiments/real/08_cooldown_atomman_cpu.sh
set -u
cd "$HOME/lerobot-experiments" || exit 1
PY=venv/bin/python
SRC=results/runs/real/apple_joint_224_r34/checkpoints/002000/pretrained_model
OUT=results/runs/real/apple_joint_224_r34/cooldown_atomman_short1k
LOG=/tmp/cd_atomman.log
[ -s "$SRC/model.safetensors" ] || { echo "ckpt 2000 absent sur atomman"; exit 1; }

CFG=/tmp/apple_cd_atomman.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"

echo "[atomman-cd $(date '+%H:%M')] START cooldown court 1k CPU (OMP=12, P-cores)" | tee "$LOG"
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 EMA=1 COOLDOWN_STEPS=1000 COOLDOWN_LR0=1e-4 \
  taskset -c 0-11 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC --policy.device=cpu \
  --dataset.video_backend=pyav \
  --output_dir=$OUT --steps=1000 --save_freq=500 --log_freq=100 --eval_freq=0 --num_workers=4 \
  >> "$LOG" 2>&1
echo "[atomman-cd $(date '+%H:%M')] DONE (rc=$?)" | tee -a "$LOG"
