#!/bin/bash
# Complète l'éval run31-jumeau : les checkpoints 28k->2k avaient été SKIPPÉS (mac2 endormi à 20:48).
# Transfère depuis mac2 (revenu) + éval n=100 descendante, append à rollouts_100.csv. En FICHIER.
# Lancer : nohup bash experiments/can/68_complete_run31jumeau_eval.sh > /tmp/run31cl_eval2.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP=100.110.237.53
RUN=results/runs/can/run31_constLR

echo "[68] $(date '+%H:%M') complète run31-jumeau 28k->2k (n=100)"
for s in 028000 026000 024000 022000 020000 018000 016000 014000 012000 010000 008000 006000 004000 002000; do
  # cède si une autre éval occupe le Principal
  while pgrep -f "37_eval_joint_birdview|13_eval_camshift|47_eval_joint_ensemble" >/dev/null; do sleep 60; done
  CK=$RUN/checkpoints/$s/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "cd ~/lerobot-experiments && tar czf - $RUN/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] && \
    $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RUN" --steps "${s#0}" --n 100 --out "$RUN/rollouts_100.csv" \
    || echo "[68] $s indisponible, skip"
done
echo "[68] DONE — convergence run31-jumeau LR constant complète :"; cut -d, -f1,4 "$RUN/rollouts_100.csv"
