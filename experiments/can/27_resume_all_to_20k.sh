#!/bin/bash
# Resume chain : reprend l'entraînement des 10 runs Mac Can sous-entraînés
# de 10 000 → 20 000 steps (+10k), puis re-éval (50 val) chaque run.
# Conserve les évals d'origine (XX_eval.json), nouvelles évals = XX_eval_20k.json.
# Lancer : nohup bash experiments/can/27_resume_all_to_20k.sh > results/logs/can/run_27_resume_all.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

# (run_dir, eval_script, train_config_subpath)
# Ordre = priorité (les plus importants en premier)
declare -a RUNS=(
  "26_proprio_birdview_resnet34|17_eval_proprio_birdview.py"
  "16_proprio_birdview|17_eval_proprio_birdview.py"
  "25_proprio_birdview_big|17_eval_proprio_birdview.py"
  "02_proprio|03_eval_proprio.py"
  "08_proprio_wrist_sep|07_eval_proprio_wrist.py"
  "14_proprio_wrist_sep_big|07_eval_proprio_wrist.py"
  "06_proprio_wrist|07_eval_proprio_wrist.py"
  "23_proprio_birdview_160|24_eval_proprio_birdview_160.py"
  "18_proprio_birdview_aug|17_eval_proprio_birdview.py"
  "21_proprio_birdview_crop|17_eval_proprio_birdview.py"
)

N_TOTAL=${#RUNS[@]}
i=0
for entry in "${RUNS[@]}"; do
  i=$((i+1))
  run=$(echo "$entry" | cut -d'|' -f1)
  eval_script=$(echo "$entry" | cut -d'|' -f2)
  run_dir="results/runs/can/$run"

  # Vérif eval déjà fait (skip)
  eval_json="results/runs/can/${run}_eval_20k.json"
  if [ -f "$eval_json" ]; then
    echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] $run déjà fait, skip"
    continue
  fi

  # Vérif checkpoint 10000 existe
  cfg="$run_dir/checkpoints/last/pretrained_model/train_config.json"
  if [ ! -f "$cfg" ]; then
    echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! $run : pas de checkpoint 10k, skip"
    continue
  fi

  echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] TRAIN RESUME $run 10k -> 20k"
  $PY -u experiments/lift/50_train_valloss.py \
    --config_path="$cfg" --resume=true \
    --output_dir="$run_dir" --steps=20000 --save_freq=2000 --log_freq=200 \
    || { echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! TRAIN RESUME ÉCHOUÉ $run (continue)" ; continue ; }

  echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] EVAL 50 val $run"

  # eval avec script approprié
  if [ "$eval_script" = "24_eval_proprio_birdview_160.py" ]; then
    $PY -u experiments/can/$eval_script --run-dir "$run_dir" --out "$eval_json" \
      || echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! EVAL ÉCHOUÉE $run"
  else
    $PY -u experiments/can/$eval_script --run-dir "$run_dir" --out "$eval_json" \
      || echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] !!! EVAL ÉCHOUÉE $run"
  fi
  echo "[27] $(date '+%H:%M:%S') [$i/$N_TOTAL] $run DONE"
done

echo "[27] $(date '+%H:%M:%S') ALL RESUME DONE"
