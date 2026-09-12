#!/bin/bash
# CELLULE MANQUANTE de la matrice : run31-jumeau (cartésien, LR CONSTANT) + SWA.
# Teste "constant + SWA ≈ cosine ?" sur le MÊME modèle cartésien. Mêmes fenêtres que le cosine
# (late5 32-40k, late10 22-40k). Construit les modèles SWA puis évalue (cartésien, n=100).
# Séquencé après [69] DONE (file Principal sans collision). En FICHIER.
# Lancer : nohup bash experiments/can/70_eval_swa_run31jumeau.sh > /tmp/swa_run31jum.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP=100.110.237.53
SRC=results/runs/can/run31_constLR

echo "[70] $(date '+%H:%M') attente [69] DONE (fin SWA-cosine)..."
while ! grep -aq "\[69\] DONE" /tmp/swa_cos_eval.log 2>/dev/null; do sleep 120; done

# s'assure que les checkpoints 22k-40k sont locaux (sinon transfert mac2)
for s in 022000 024000 026000 028000 030000 032000 034000 036000 038000 040000; do
  CK=$SRC/checkpoints/$s/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "cd ~/lerobot-experiments && tar czf - $SRC/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
done

echo "[70] construit modèles SWA constant (late5, late10)..."
$PY experiments/can/65_swa_make.py --src $SRC --steps 32000,34000,36000,38000,40000 \
  --out results/runs/can/run31jum_swa_const_late5 2>&1 | tail -1
$PY experiments/can/65_swa_make.py --src $SRC --steps 22000,24000,26000,28000,30000,32000,34000,36000,38000,40000 \
  --out results/runs/can/run31jum_swa_const_late10 2>&1 | tail -1

# garde anti-collision : Principal libre
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift|47_eval_joint_ensemble" >/dev/null; do sleep 60; done
echo "[70] $(date '+%H:%M') Principal libre -> éval SWA-constant (cartésien, n=100)"

for name in const_late5 const_late10; do
  RD=results/runs/can/run31jum_swa_$name
  [ -s "$RD/checkpoints/000000/pretrained_model/model.safetensors" ] || { echo "[70] $name absent"; continue; }
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RD" --steps 0 --n 100 --out "$RD/r.csv"
done

echo "[70] DONE — MATRICE schedule x SWA (cartésien) :"
echo "  cosine brut (run31)         : 0.948 (@500)"
echo "  constant brut (run31-jumeau): ~0.50 (@40k)"
for name in const_late5 const_late10; do
  v=$(cut -d, -f4 results/runs/can/run31jum_swa_$name/r.csv 2>/dev/null | tail -1)
  echo "  constant + SWA ($name) : $v"
done
echo "  (cosine + SWA : voir /tmp/swa_cos_eval.log)"
