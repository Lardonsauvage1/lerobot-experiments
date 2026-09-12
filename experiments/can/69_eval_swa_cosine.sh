#!/bin/bash
# Teste le SWA sur le modèle COSINE (run31, 94,8%) : moyenne des checkpoints cosine tardifs.
# Hypothèse : le cosine a déjà "posé" le modèle (LR->0) -> SWA quasi neutre (≈94,8%). Si > 94,8% -> SWA
# aide partout. Évalue cos_late5 (32-40k) et cos_late10 (22-40k), cartésien n=100.
# Séquencé après [67] DONE pour ne pas entrer en collision (file Principal). En FICHIER.
# Lancer : nohup bash experiments/can/69_eval_swa_cosine.sh > /tmp/swa_cos_eval.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

echo "[69] $(date '+%H:%M') attente [67] DONE (fin SWA+ensembling joint)..."
while ! grep -aq "\[67\] DONE" /tmp/swa_ens.log 2>/dev/null; do sleep 120; done
# garde anti-collision : Principal libre
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift|47_eval_joint_ensemble" >/dev/null; do sleep 60; done
echo "[69] $(date '+%H:%M') Principal libre -> éval SWA-cosine (cartésien, n=100)"

for name in cos_late5 cos_late10; do
  RD=results/runs/can/run31_swa_$name
  [ -s "$RD/checkpoints/000000/pretrained_model/model.safetensors" ] || { echo "[69] $name absent"; continue; }
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RD" --steps 0 --n 100 --out "$RD/r.csv"
done

echo "[69] DONE — SWA sur COSINE (run31, baseline cosine = 94,8%@500) :"
for name in cos_late5 cos_late10; do
  v=$(cut -d, -f4 results/runs/can/run31_swa_$name/r.csv 2>/dev/null | tail -1)
  echo "  $name : $v"
done
