#!/bin/bash
# Test accélération atomman : p8 et p6 (base22 + p12 déjà mesurés).
# Lance chaque config en resume, 240s, mesure s/step, tue. État reste à 10500.
cd "/home/sam/Documents/projet VScode/experience_Le" || exit 1
CFG="results/runs/phase5_methodology/26_resnet34_dense_continue/checkpoints/last/pretrained_model/train_config.json"
RES=/tmp/accel_results.txt
: > "$RES"

run_test() {
  TAG="$1"; OMP="$2"; CPUS="$3"
  pkill -9 -f 50_train_valloss 2>/dev/null; sleep 3
  OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" nohup taskset -c "$CPUS" \
    venv312/bin/python -u experiments/lift/50_train_valloss.py \
    --config_path="$CFG" --resume=true > "/tmp/accel_$TAG.log" 2>&1 &
  PID=$!
  sleep 240
  VALS=$(tr '\r' '\n' < "/tmp/accel_$TAG.log" | grep -aoE '[0-9.]+s/step' | tail -6 | tr '\n' ' ')
  echo "$TAG (OMP=$OMP cpu=$CPUS) : $VALS" >> "$RES"
  kill -9 $PID 2>/dev/null; pkill -9 -f 50_train_valloss; sleep 3
}

run_test p8 8 0-7
run_test p6 6 0,1,3,6,8,10
echo "BATCH_DONE" >> "$RES"
