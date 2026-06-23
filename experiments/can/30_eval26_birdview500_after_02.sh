#!/bin/bash
# Attend la fin de la reprise de 02_proprio (signal = 02_proprio_eval_20k.json écrit en toute fin),
# puis lance l'éval 500 rollouts du birdview ResNet34 (run 26, ajouté à MODELS de 10_vision_500_rollouts.py).
# Le script 10 skippe les modèles déjà dans vision_500.json -> seul 26 sera évalué (même 500 états figés -> comparable).
# Lancer : nohup bash experiments/can/30_eval26_birdview500_after_02.sh > results/logs/can/run_30_eval26_500.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
SIGNAL=results/runs/can/02_proprio_eval_20k.json

echo "[30] $(date '+%H:%M:%S') attente fin de 02 (signal: $SIGNAL)"
until [ -f "$SIGNAL" ]; do sleep 60; done
echo "[30] $(date '+%H:%M:%S') 02 terminé -> éval 500 rollouts birdview ResNet34 (run 26)"

$PY -u experiments/can/10_vision_500_rollouts.py
echo "[30] $(date '+%H:%M:%S') DONE — résultat dans results/runs/can/vision_500.json"
