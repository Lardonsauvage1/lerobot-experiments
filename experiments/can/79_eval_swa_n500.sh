#!/bin/bash
# Réévaluation n=500 des fenêtres rivière W1(54), v21(40), late10(88) : resserrer l'IC95 (±4 vs ±12)
# pour confirmer que la montée 40-54 -> 88 est réelle. Puis régénère le graphe.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
echo "[79] $(date '+%H:%M') attente Principal libre..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|47_eval_joint_ensemble" >/dev/null; do sleep 60; done
for w in W1_12_20k v21_16_24k late10; do
  RD=results/runs/can/joint_swa_$w
  echo "[79] $(date '+%H:%M') éval n=500 : $w"
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RD" --steps 0 --n 500 --kp 50 --out "$RD/r.csv" || echo "[79] FAIL $w"
done
$PY experiments/can/77_plot_river.py 2>&1 | tail -1
echo "[79] DONE — n=500 :"
for w in W1_12_20k v21_16_24k late10; do echo "  $w : $(cut -d, -f4,5,6 results/runs/can/joint_swa_$w/r.csv|tail -1)"; done
