#!/bin/bash
# PROFIL DE LA RIVIÈRE : évalue le FOND (SWA) à des fenêtres successives de l'entraînement joint
# (12-20k, 22-30k, 32-40k, 42-50k). Si le fond MONTE puis plafonne -> le LR constant fait progresser
# globalement (hypothèse WSD/SWA valide chez nous). Si plat -> le constant rebondit sans avancer.
# Séquencé après le pipeline maître (TOUT FINI) pour éviter toute collision MPS. En FICHIER.
# Lancer : nohup bash experiments/can/73_river_profile.sh > /tmp/river.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

echo "[73] $(date '+%H:%M') attente fin du pipeline maître (PHASE 4 DONE)..."
while ! grep -aq "PHASE 4 DONE\|TOUT FINI" /tmp/master.log 2>/dev/null; do sleep 180; done
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift|47_eval_joint_ensemble|50_train_valloss" >/dev/null; do sleep 60; done
echo "[73] $(date '+%H:%M') Principal libre -> profil de la rivière (fonds SWA, n=50)"

for w in W1_12_20k W2_22_30k W3_32_40k W4_42_50k; do
  RD=results/runs/can/joint_swa_$w
  [ -s "$RD/checkpoints/000000/pretrained_model/model.safetensors" ] || { echo "[73] $w absent"; continue; }
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RD" --steps 0 --n 50 --kp 50 --out "$RD/r.csv" || echo "[73] FAIL $w"
done

echo "[73] DONE — PROFIL DE LA RIVIÈRE (fond SWA par fenêtre) :"
for w in W1_12_20k W2_22_30k W3_32_40k W4_42_50k; do
  v=$(cut -d, -f4 results/runs/can/joint_swa_$w/r.csv 2>/dev/null | tail -1)
  echo "  $w : fond = $v"
done
echo "  -> si le fond monte (W1<...<W4) puis plafonne = le LR constant fait progresser globalement (WSD/SWA valide)"
