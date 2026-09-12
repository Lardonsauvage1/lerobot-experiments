#!/bin/bash
# Rollouts n=100 par checkpoint du run31-jumeau (LR constant), en ordre DESCENDANT (40k d'abord, 2k en dernier).
# Sur le Principal (robosuite). Attend la fin de l'entraînement run31-jumeau sur mac2, puis transfère
# chaque checkpoint depuis mac2 et l'évalue (32_eval_dense_birdview, clean, n=100).
# En FICHIER. nohup caffeinate -i bash experiments/can/54_eval_run31_constLR_desc.sh > /tmp/run31cl_eval.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
IP=100.110.237.53
RUN=results/runs/can/run31_constLR
OUT=$RUN/rollouts_100.csv
STEPS="40000 38000 36000 34000 32000 30000 28000 26000 24000 22000 20000 18000 16000 14000 12000 10000 8000 6000 4000 2000"

echo "[54] $(date '+%H:%M') attente fin entraînement run31-jumeau sur mac2..."
# attend que le training existe ET soit fini (process parti + checkpoint 40000 présent)
while true; do
  done_flag=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*run31_constLR' >/dev/null && echo RUN || (ls ~/lerobot-experiments/$RUN/checkpoints/040000 >/dev/null 2>&1 && echo DONE || echo WAIT)" 2>/dev/null)
  [ "$done_flag" = "DONE" ] && break
  sleep 300
done
echo "[54] $(date '+%H:%M') entraînement fini -> rollouts n=100 DESCENDANT (40k->2k)"

for s in $STEPS; do
  pad=$(printf "%06d" "$s")
  CK=$RUN/checkpoints/$pad/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RUN/checkpoints/$pad/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] || { echo "[54] $pad indisponible, skip"; continue; }
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RUN" --steps "$s" --n 100 --out "$OUT"
done
echo "[54] DONE — rollouts run31-jumeau LR constant :"; cat "$OUT"
