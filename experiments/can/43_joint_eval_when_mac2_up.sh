#!/bin/bash
# REPRISE AUTOMATIQUE de l'éval joint @500 dès que mac2 redevient joignable.
# Contexte : mac2 s'est endormi à la fin du training (caffeinate terminé). On a déjà
# model.safetensors localement MAIS pas les normaliseurs (.safetensors séparés) ni les
# configs -> éval impossible sans eux. Ce script attend mac2, transfère le checkpoint
# COMPLET (modèle + normaliseurs + 4 configs), vérifie l'intégrité, puis lance 37 @500.
#
#   caffeinate -i bash experiments/can/43_joint_eval_when_mac2_up.sh
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
IP=100.110.237.53
RD=results/runs/can/joint_r34_bigunet
REM=results/runs/can/joint_r34_bigunet/checkpoints/040000/pretrained_model
OUT=$RD/rollouts_500.csv

echo "[43] $(date '+%H:%M %d/%m') — attente du retour de mac2 (poll 180s)..."
i=0
while true; do
  "$TS" ping -c 1 "$IP" >/dev/null 2>&1
  if ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 'true' 2>/dev/null; then
    echo "[43] $(date '+%H:%M') — mac2 JOIGNABLE après ${i} essais ✓"
    break
  fi
  i=$((i+1)); sleep 180
done

echo "[43] transfert du checkpoint COMPLET (modèle + normaliseurs + configs)..."
for try in 1 2 3 4 5; do
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $REM" | tar xzf - -C . \
    && break
  echo "[43] tentative $try échouée, retry dans 30s..."; sleep 30
done

# Intégrité : il faut les 7 fichiers (modèle + 2 normaliseurs .safetensors + 4 JSON)
NEED=(model.safetensors config.json train_config.json policy_preprocessor.json \
      policy_postprocessor.json policy_preprocessor_step_3_normalizer_processor.safetensors \
      policy_postprocessor_step_0_unnormalizer_processor.safetensors)
miss=0
for f in "${NEED[@]}"; do
  [ -s "$REM/$f" ] || { echo "[43] MANQUE: $f"; miss=1; }
done
if [ $miss -ne 0 ]; then echo "[43] ⚠️ transfert incomplet — éval NON lancée, voir ci-dessus"; exit 1; fi
echo "[43] checkpoint complet ✓ — lancement éval JOINT @500"

venv312/bin/python -u experiments/can/37_eval_joint_birdview.py \
  --run-dir "$RD" --steps 40000 --n 500 --kp 50 --out "$OUT"

echo "[43] ÉVAL JOINT @500 TERMINÉE — résultat:"; cat "$OUT"
echo "[43] DONE"
