#!/bin/bash
# Courbe de convergence JOINT (n=50 par checkpoint), checkpoints du PLUS ENTRAÎNÉ au moins entraîné.
# Pour chaque step (décroissant) : rapatrie le checkpoint depuis mac2 si absent, puis évalue (37).
# 37 ajoute à rollouts_50.csv (clé = step, skip si déjà fait) -> on peut interrompre, les points
# importants (les plus entraînés) sont obtenus en premier.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
IP="${MAC2_HOST:-mac2}"
RUN=results/runs/can/joint_r34_bigunet
OUT=$RUN/rollouts_50.csv
STEPS="40000 38000 36000 34000 32000 30000 28000 26000 24000 22000 20000 18000 16000 14000 12000 10000 8000 6000 4000 2000"

for s in $STEPS; do
  pad=$(printf "%06d" "$s")
  CK=$RUN/checkpoints/$pad/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    echo "[14] $(date '+%H:%M') transfert checkpoint $pad depuis mac2..."
    for try in 1 2 3; do
      "$TS" ping -c 2 "$IP" >/dev/null 2>&1
      if ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RUN/checkpoints/$pad/pretrained_model" | tar xzf - -C . ; then break; fi
      echo "[14] retry transfert $pad ($try)..."; sleep 20
    done
  fi
  if [ ! -s "$CK/model.safetensors" ]; then echo "[14] ⚠️ $pad indisponible, skip"; continue; fi
  $PY -u experiments/can/37_eval_joint_birdview.py \
    --run-dir "$RUN" --steps "$s" --n 50 --kp 50 --out "$OUT"
done
echo "[14] CONVERGENCE JOINT TERMINÉE"; cat "$OUT"
