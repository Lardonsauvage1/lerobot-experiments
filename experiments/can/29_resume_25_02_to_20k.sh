#!/bin/bash
# Reprise ciblée : runs 3 et 4 de la chaîne 27 uniquement (les autres mis de côté).
#   3) 25_proprio_birdview_big  -> checkpoints supprimes : NON reprenable (skip, signale)
#   4) 02_proprio                -> reprend 10k -> 20k puis re-eval (50 val)
# Lancer : nohup bash experiments/can/29_resume_25_02_to_20k.sh > results/logs/can/run_29_resume_25_02.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

declare -a RUNS=(
  "25_proprio_birdview_big|17_eval_proprio_birdview.py"
  "02_proprio|03_eval_proprio.py"
)

N_TOTAL=${#RUNS[@]}
i=0
for entry in "${RUNS[@]}"; do
  i=$((i+1))
  run=$(echo "$entry" | cut -d'|' -f1)
  eval_script=$(echo "$entry" | cut -d'|' -f2)
  run_dir="results/runs/can/$run"

  eval_json="results/runs/can/${run}_eval_20k.json"
  if [ -f "$eval_json" ]; then
    echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] $run déjà fait, skip"
    continue
  fi

  cfg="$run_dir/checkpoints/last/pretrained_model/train_config.json"
  if [ ! -f "$cfg" ]; then
    echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! $run : pas de checkpoint (dir absent ou nettoyé), skip"
    continue
  fi

  echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] TRAIN RESUME $run 10k -> 20k"
  $PY -u experiments/lift/50_train_valloss.py \
    --config_path="$cfg" --resume=true \
    --output_dir="$run_dir" --steps=20000 --save_freq=2000 --log_freq=200 \
    || { echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! TRAIN RESUME ÉCHOUÉ $run (continue)" ; continue ; }

  echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] EVAL 50 val $run"
  $PY -u experiments/can/$eval_script --run-dir "$run_dir" --out "$eval_json" \
    || echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! EVAL ÉCHOUÉE $run"
  echo "[29] $(date '+%H:%M:%S') [$i/$N_TOTAL] $run DONE"
done

echo "[29] $(date '+%H:%M:%S') RESUME 25/02 DONE"
