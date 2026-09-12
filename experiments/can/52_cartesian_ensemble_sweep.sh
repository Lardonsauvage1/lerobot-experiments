#!/bin/bash
# Teste l'ensembling sur le CARTÉSIEN (run31) à 24k/30k/40k, baseline vs ensemble (m=0.01, n=30).
# But : l'ensembling aide-t-il aussi le cartésien (déjà ~94,8% / déjà lissé par l'OSC) ?
# En FICHIER (pas de self-match). À lancer quand le Principal est libre (après E_joint50).
# nohup caffeinate -i bash experiments/can/52_cartesian_ensemble_sweep.sh > /tmp/cart_sweep.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
OUT=results/runs/can/31_proprio_birdview_r34_bigunet/cartesian_ensemble.csv

for STEP in 24000 30000 40000; do
  echo "[52] === step=$STEP baseline ==="
  $PY -u experiments/can/48_eval_cartesian_ensemble.py --steps $STEP --n 30 --no-ensemble --out "$OUT"
  echo "[52] === step=$STEP ensemble m=0.01 ==="
  $PY -u experiments/can/48_eval_cartesian_ensemble.py --steps $STEP --n 30 --m 0.01 --out "$OUT"
done
echo "[52] DONE — tableau cartésien baseline vs ensemble :"; cat "$OUT"
