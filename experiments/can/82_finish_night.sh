#!/bin/bash
# Reprise après disque-plein : finit le cooldown (4k/5k, décisifs) en libérant l'espace,
# puis lance la rivière mini-CNN. Tout sur Principal. En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/82_finish_night.sh > /tmp/finish.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/joint_cooldown_50k

echo "[82] $(date '+%H:%M') éval cooldown 4k/5k (LR -> 0, points décisifs vs SWA 81,6 %)"
for s in 4000 5000; do
  d=$(printf '%06d' "$s")
  if [ -s "$RUN/checkpoints/$d/pretrained_model/model.safetensors" ]; then
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RUN" --steps "$s" --n 50 --kp 50 --out "$RUN/rollouts_50.csv"
  fi
done

echo "[82] $(date '+%H:%M') cooldown terminé. Résultats :"
cat "$RUN/rollouts_50.csv"

echo "[82] nettoyage des checkpoints cooldown (résultats dans le CSV) -> libère ~1,4 Go"
rm -rf "$RUN/checkpoints"
sync; df -h /System/Volumes/Data | tail -1

echo "[82] $(date '+%H:%M') -> rivière mini-CNN cos-vs-const"
bash experiments/phase5_methodology/33_river_minicnn_cos_vs_const.sh

echo "[82] $(date '+%H:%M') DONE"
