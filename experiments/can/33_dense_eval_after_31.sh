#!/bin/bash
# Après la fin de l'entraînement run 31 (R34 + gros U-Net), éval DENSE 50 rollouts
# par checkpoint -> rollouts_50.csv (trajectoire de convergence), puis régénère l'étude.
# L'éval (CPU parallèle) ne démarre qu'une fois l'entraînement terminé (libère le Mac).
# Lancer : nohup bash experiments/can/33_dense_eval_after_31.sh > results/logs/can/run_33_dense_after_31.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/31_proprio_birdview_r34_bigunet
LOG=results/logs/can/run_31_r34_bigunet.log

echo "[33] $(date '+%H:%M:%S') attente fin entraînement run 31 (TRAIN DONE)"
until grep -q "TRAIN DONE" "$LOG" 2>/dev/null; do sleep 120; done
echo "[33] $(date '+%H:%M:%S') entraînement fini -> éval dense 50 rollouts/checkpoint"

$PY -u experiments/can/32_eval_dense_birdview.py \
  --run-dir "$RUN" --steps "" --n 50 \
  --out "$RUN/rollouts_50.csv" \
  || { echo "[33] $(date '+%H:%M:%S') ÉVAL DENSE ÉCHEC -> arrêt chaîne" ; exit 1 ; }

echo "[33] $(date '+%H:%M:%S') régénération de l'étude de convergence"
$PY experiments/phase5_methodology/33_plot_convergence_rollouts.py

echo "[33] $(date '+%H:%M:%S') benchmark perf 4 conditions (sortie -> run_34_benchmark.log)"
$PY -u experiments/can/34_bench_compile_mlx.py > results/logs/can/run_34_benchmark.log 2>&1 \
  || echo "[33] benchmark a retourné une erreur (à corriger au réveil)"

echo "[33] $(date '+%H:%M:%S') DONE"
