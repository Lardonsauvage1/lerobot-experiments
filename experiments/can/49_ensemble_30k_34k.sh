#!/bin/bash
# Temporal ensembling (m=0.01, n=30) sur les checkpoints 30k (meilleur, 56% sans ensemble) et
# 34k (pire, 0% sans ensemble), pour voir si l'ensembling est un correcteur UNIVERSEL.
# Attend la fin de E_lookat (13_eval_camshift_lookat) pour ne pas se battre sur le MPS du Principal.
# En FICHIER -> pgrep ne matche pas sa propre ligne (pas de bug self-match).
# Lancer : nohup caffeinate -i bash experiments/can/49_ensemble_30k_34k.sh > /tmp/ens_30_34.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
OUT=results/runs/can/joint_r34_bigunet/ensemble_eval.csv

echo "[49] $(date '+%H:%M') attente fin E_lookat (13_eval_camshift_lookat)..."
while pgrep -f "13_eval_camshift_lookat" >/dev/null; do sleep 60; done
echo "[49] $(date '+%H:%M') Principal libre -> ensemble 30k puis 34k (m=0.01, n=30)"

for STEP in 30000 34000; do
  echo "[49] === ensemble step=$STEP ==="
  $PY -u experiments/can/47_eval_joint_ensemble.py --steps $STEP --n 30 --m 0.01 --out "$OUT"
done
echo "[49] DONE — tableau ensemble :"; cat "$OUT"
