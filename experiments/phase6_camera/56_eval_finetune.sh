#!/bin/bash
# Éval du fine-tune look-at (warm-start run31) : transfère ses checkpoints depuis mac2, puis
# (1) VERDICT camshift n=100 sur le checkpoint final ; (2) convergence clean n=100 par checkpoint.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
RD=results/runs/phase6_camera/lookat_finetune_run31
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP=100.110.237.53
PY=venv312/bin/python

echo "[56] $(date '+%H:%M') transfert checkpoints fine-tune depuis mac2..."
for s in 002000 004000 006000 008000 010000 012000 014000 015000; do
  CK=$RD/checkpoints/$s/pretrained_model
  [ -s "$CK/model.safetensors" ] && continue
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RD/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
done

echo "[56] === VERDICT : camshift robustesse n=100 sur le checkpoint FINAL (15k) ==="
$PY -u experiments/phase6_camera/13_eval_camshift_lookat.py \
  --ckpt $RD/checkpoints/015000/pretrained_model \
  --levels 0:0,2:2,5:5,10:10,15:15,20:20 --n 100 --out $RD/camshift.csv

echo "[56] === CONVERGENCE : clean n=100 par checkpoint (descendant) ==="
$PY -u experiments/can/32_eval_dense_birdview.py --run-dir $RD \
  --steps "15000,14000,12000,10000,8000,6000,4000,2000" --n 100 --out $RD/rollouts_100.csv

echo "[56] DONE — camshift fine-tune :"; cat $RD/camshift.csv
