#!/bin/bash
# B — évalue le joint ResNet34 PRÉ-ENTRAÎNÉ ImageNet (entraîné 40k le 27/06, jamais évalué).
# Rapatrie les checkpoints depuis mac2, évalue la convergence brute (n=50, tous les 8k),
# supprime chaque checkpoint après éval (économie disque). Compare au from-scratch.
# Lancer : nohup caffeinate -i bash experiments/can/94_eval_pretrained.sh > /tmp/eval_pretrained.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/joint_r34_pretrained
OUT=$RUN/rollouts_50.csv
STEPS="8000 16000 24000 32000 40000"   # convergence brute, 5 points

echo "[94] $(date '+%H:%M') éval convergence B (pré-entraîné) — rollouts bruts n=50"
mkdir -p "$RUN/checkpoints"
for s in $STEPS; do
  d=$(printf '%06d' $s)
  grep -q "^${s%0*}" "$OUT" 2>/dev/null && grep -q ",$s," "$OUT" 2>/dev/null
  # garde-fou disque
  while [ "$(df -m . | awk 'NR==2{print $4}')" -lt 4000 ]; do echo "[94] ⏸ disque < 4Go, pause 3min"; sleep 180; done
  # rapatrier depuis mac2 si absent
  CK="$RUN/checkpoints/$d/pretrained_model"
  if [ ! -s "$CK/model.safetensors" ]; then
    echo "[94] tire ${s} depuis mac2"
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RUN/checkpoints/$d/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] || { echo "[94] ${s}k absent, skip"; continue; }
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RUN" --steps "$s" --n 50 --kp 50 --out "$OUT"
  echo "[94] ${s}k -> $(tail -1 "$OUT" | cut -d, -f4)"
  rm -rf "$RUN/checkpoints/$d"   # libère ~700 Mo après éval
done

echo "[94] $(date '+%H:%M') DONE — convergence B :"
cat "$OUT"
echo "[94] comparaison from-scratch (joint_r34_bigunet 40k brut) : ~12%"