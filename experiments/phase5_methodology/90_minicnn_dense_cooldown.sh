#!/bin/bash
# mini-CNN CONSTANT — profil COOLDOWN + MERGE, grille 5k (0-100k). n=500.
#   Principal : build+éval les MERGES (SWA 5 ckpts centrés) ; tire+évalue les COOLDOWNS de mac2 (91).
# Attend la fin du joint dense (88) + Principal libre. Reprenable. En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/phase5_methodology/90_minicnn_dense_cooldown.sh > /tmp/mini_dense.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
SRC=results/runs/phase5_methodology/mini_constant_all
CPROF=results/runs/can/cooldown_profile_minicnn.csv
MPROF=results/runs/can/merge_profile_minicnn.csv
N=200
# watchdog : fonction (pas une variable -> les quotes perl sont préservées). Tue toute éval > 1200 s
# (anti-deadlock robosuite/multiprocessing macOS) ; 8 workers = plus stable.
run_eval() {  # args passés à 12_eval_parallel ; renvoie le code de sortie de l'éval
  perl -e 'alarm 1200; exec @ARGV' $PY -u experiments/phase5_methodology/12_eval_parallel.py \
    --workers 8 --max-steps 200 --infer-steps 4 --n $N "$@"
}
valid_row() { [ -s "$1" ] && [ -n "$(tail -1 "$1" | cut -d, -f4)" ]; }  # r.csv a un success_rate
[ -s "$CPROF" ] || echo "source_k,success_rate,ci95_low,ci95_high,n" > "$CPROF"
[ -s "$MPROF" ] || echo "source_k,success_rate,ci95_low,ci95_high,n" > "$MPROF"

echo "[90] $(date '+%H:%M') attente fin joint dense (88) + Principal libre..."
while pgrep -f "37_eval_joint_birdview|50_train_valloss|12_eval_parallel" >/dev/null; do sleep 60; done
echo "[90] $(date '+%H:%M') -> merges + cooldowns mini-CNN"

for K in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  # --- MERGE (local) : SWA 5 ckpts centrés 5k-espacés ---
  if ! grep -q "^${K}," "$MPROF"; then
    LO=$(( K-10 < 5 ? 5 : K-10 ))
    STEPS=$(printf '%d000,%d000,%d000,%d000,%d000' $LO $((LO+5)) $((LO+10)) $((LO+15)) $((LO+20)))
    OUT=results/runs/phase5_methodology/mini_merge_${K}k
    $PY experiments/can/65_swa_make.py --src "$SRC" --steps "$STEPS" --out "$OUT" 2>&1 | tail -1
    if [ -s "$OUT/checkpoints/000000/pretrained_model/model.safetensors" ]; then
      run_eval --run-dir "$OUT" --steps 0 --out "$OUT/r.csv"
      if valid_row "$OUT/r.csv"; then
        row=$(tail -1 "$OUT/r.csv"); echo "${K},$(echo "$row"|cut -d, -f4),$(echo "$row"|cut -d, -f5),$(echo "$row"|cut -d, -f6),${N}" >> "$MPROF"
        echo "[90] merge ${K}k -> $(echo "$row"|cut -d, -f4)"
      else echo "[90] merge ${K}k ÉCHEC éval (non appendé, retry au prochain run)"; fi
    fi
    rm -rf "$OUT/checkpoints"
  fi
  # --- COOLDOWN (tire de mac2) ---
  if ! grep -q "^${K}," "$CPROF"; then
    STEP=$((K*1000+5000)); D=$(printf '%06d' $STEP)
    OUT=results/runs/phase5_methodology/mini_cd_${K}k
    REMOTE="results/runs/phase5_methodology/mini_cd_${K}k/checkpoints/$D/pretrained_model/model.safetensors"
    echo "[90] $(date '+%H:%M') attente cooldown mac2 ${K}k..."
    t=0; until ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null; do
      t=$((t+1)); [ $t -ge 25 ] && { echo "[90] ${K}k cooldown indispo mac2 (skip, retry au prochain run)"; break; }; sleep 60; done
    ssh -o BatchMode=yes mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null || continue
    mkdir -p "$OUT/checkpoints"
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - results/runs/phase5_methodology/mini_cd_${K}k/checkpoints/$D/pretrained_model" | tar xzf - -C . 2>/dev/null
    run_eval --run-dir "$OUT" --steps "$STEP" --out "$OUT/r.csv"
    if valid_row "$OUT/r.csv"; then
      row=$(tail -1 "$OUT/r.csv"); echo "${K},$(echo "$row"|cut -d, -f4),$(echo "$row"|cut -d, -f5),$(echo "$row"|cut -d, -f6),${N}" >> "$CPROF"
      echo "[90] cooldown ${K}k -> $(echo "$row"|cut -d, -f4)"
      rm -rf "$OUT/checkpoints"; ssh -o BatchMode=yes mac2 "rm -rf ~/lerobot-experiments/results/runs/phase5_methodology/mini_cd_${K}k" 2>/dev/null
    else
      echo "[90] cooldown ${K}k ÉCHEC éval -> checkpoint mac2 CONSERVÉ (retry au prochain run)"
      rm -rf "$OUT/checkpoints"   # libère le local, garde mac2
    fi
  fi
done

echo "[90] $(date '+%H:%M') DONE. Graphe :"
$PY experiments/phase5_methodology/92_plot_minicnn_cooldown.py 2>&1 | tail -1
echo "[90] cooldown :"; cat "$CPROF"
echo "[90] merge :"; cat "$MPROF"
