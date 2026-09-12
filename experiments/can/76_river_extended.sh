#!/bin/bash
# Profil rivière DENSIFIÉ (vérifier que ce n'est pas 4 points chanceux) + ÉTENDU 50k->80k
# (le fond continue-t-il de monter ? peut-on atteindre 88% avec une fenêtre large ?).
# Évalue 7 fenêtres SWA (n=50) APRÈS l'éval EMA (75). En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/76_river_extended.sh > /tmp/river_ext.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
WINS="v21_16_24k v31_26_34k v41_36_44k e56_52_60k e66_62_70k e76_72_80k wide_42_80k"

echo "[76] $(date '+%H:%M') attente fin éval EMA (75) + Principal libre..."
while pgrep -f "75_eval_push_ema" >/dev/null; do sleep 120; done
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview" >/dev/null; do sleep 60; done
echo "[76] $(date '+%H:%M') Principal libre -> éval fenêtres rivière (densifiée + étendue)"

for w in $WINS; do
  RD=results/runs/can/joint_swa_$w
  [ -s "$RD/checkpoints/000000/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RD" --steps 0 --n 50 --kp 50 --out "$RD/r.csv" \
    || echo "[76] $w absent/FAIL"
done

echo "[76] DONE — PROFIL RIVIÈRE ÉTENDU (fond par fenêtre) :"
echo "  [déjà connus] W1_12-20=54 W2_22-30=70 W3_32-40=70 W4_42-50=76 ; late10_32-50=88"
for w in $WINS; do
  echo "  $w = $(cut -d, -f4 results/runs/can/joint_swa_$w/r.csv 2>/dev/null|tail -1)"
done
