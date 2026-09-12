#!/bin/bash
# EMPILE le temporal ensembling SUR le modèle SWA late10 (88%) : les 2 techniques cumulées.
# Évalue le MÊME modèle SWA late10 sans ensemble (baseline) PUIS avec ensemble (m=0.01), n=50, mêmes états.
# Séquencé après [61] DONE (fin éval fine-tune prolongé) → tourne dans le trou Principal avant l'éval
# joint poussé (63). En FICHIER.
# Lancer : nohup bash experiments/can/67_swa_late10_ensemble.sh > /tmp/swa_ens.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
CK=results/runs/can/joint_swa_late10/checkpoints/000000/pretrained_model
BASE=results/runs/can/joint_swa_late10/baseline.csv
ENS=results/runs/can/joint_swa_late10/ensemble.csv

[ -s "$CK/model.safetensors" ] || { echo "[67] ⚠️ modèle SWA late10 absent"; exit 1; }

echo "[67] $(date '+%H:%M') attente [61] DONE (fin éval fine-tune prolongé)..."
while ! grep -aq "\[61\] DONE" /tmp/ext_ft_eval.log 2>/dev/null; do sleep 120; done
# garde anti-collision : Principal libre de toute autre éval
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift|48_eval_cartesian" >/dev/null; do sleep 60; done
echo "[67] $(date '+%H:%M') Principal libre -> SWA late10 : baseline (sans) puis +ensemble"

$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CK" --no-ensemble --steps 0 --n 50 --kp 50 --out "$BASE"
$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CK" --m 0.01     --steps 0 --n 50 --kp 50 --out "$ENS"

echo "[67] DONE — SWA late10 :"
echo "  baseline (sans ensemble) :"; cut -d, -f5 "$BASE" 2>/dev/null | tail -1
echo "  + temporal ensembling    :"; cut -d, -f5 "$ENS" 2>/dev/null | tail -1
