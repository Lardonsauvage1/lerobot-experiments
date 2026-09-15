#!/bin/bash
# Éval du push joint BRUT vs EMA (54-80k, n=50, kp=50). Manuel car P4 buggé (attendait 80000/80000).
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RDJ=results/runs/can/joint_r34_bigunet
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP="${MAC2_HOST:-mac2}"
for s in 054000 058000 062000 066000 070000 074000 078000 080000; do
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RDJ/checkpoints/$s/pretrained_model ${RDJ}_ema/checkpoints/$s/pretrained_model 2>/dev/null" | tar xzf - -C . 2>/dev/null
  [ -s "$RDJ/checkpoints/$s/pretrained_model/model.safetensors" ] && $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RDJ" --steps "${s#0}" --n 50 --kp 50 --out "$RDJ/rollouts_50.csv"
  [ -s "${RDJ}_ema/checkpoints/$s/pretrained_model/model.safetensors" ] && $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "${RDJ}_ema" --steps "${s#0}" --n 50 --kp 50 --out "${RDJ}_ema/rollouts_50.csv"
done
echo "[75] DONE"
echo "BRUT 54-80k:"; cut -d, -f1,4 "$RDJ/rollouts_50.csv" | tail -8
echo "EMA 54-80k:"; cut -d, -f1,4 "${RDJ}_ema/rollouts_50.csv" | tail -8
