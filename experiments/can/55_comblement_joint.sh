#!/bin/bash
# Comble les checkpoints 18-24k manquants de la convergence joint (n=50), dans un creux du Principal.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
RD=results/runs/can/joint_r34_bigunet
for s in 18000 20000 22000 24000; do
  pad=$(printf "%06d" "$s")
  CK=$RD/checkpoints/$pad/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    /Applications/Tailscale.app/Contents/MacOS/Tailscale ping -c 2 ${MAC2_HOST:-mac2} >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RD/checkpoints/$pad/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] || { echo "[55] $pad absent, skip"; continue; }
  venv312/bin/python -u experiments/can/37_eval_joint_birdview.py --run-dir "$RD" --steps "$s" --n 50 --kp 50 --out "$RD/rollouts_50.csv"
done
echo "[55] COMBLEMENT DONE"
