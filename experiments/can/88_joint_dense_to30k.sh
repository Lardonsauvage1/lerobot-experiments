#!/bin/bash
# JOINT — densification du profil COOLDOWN + MERGE, tous les 4k jusqu'à 30k.
# Grille {2,4,8,12,16,20,24,28,30k}. Cooldown LR=0 (5k) ET merge SWA (5 ckpts centrés) à chaque point.
#   Principal : cooldowns OWN={8,16,24} (train+éval) + TOUS les merges + tire mac2 COOLD_MAC2={2,4,12,28}.
#   mac2 (89_joint_dense_mac2.sh) : entraîne COOLD_MAC2 ; Principal pull+éval.
# Résultats -> cooldown_profile.csv (cooldown) + merge_profile.csv (merge). n=100. Reprenable. En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/88_joint_dense_to30k.sh > /tmp/dense_joint.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
SRC_RUN=results/runs/can/joint_r34_bigunet
PROF=results/runs/can/cooldown_profile.csv
MPROF=results/runs/can/merge_profile.csv
N=100
COOLD_OWN=""; COOLD_MAC2="2 4 12 16 24 28"; MERGE_PTS="2 4 8 12 16 20 24 28 30"
# garde-fou disque : tirer/évaluer un checkpoint joint écrit ~700 Mo ; pause si < 6 Go libre
wait_for_disk() {
  while [ "$(df -m . | awk 'NR==2{print $4}')" -lt 6000 ]; do
    echo "[88] $(date '+%H:%M') ⏸ disque < 6 Go libre -> pause 5 min"; sleep 300
  done
}
[ -s "$PROF" ]  || echo "source_k,success_rate,ci95_low,ci95_high,n" > "$PROF"
[ -s "$MPROF" ] || echo "source_k,success_rate,ci95_low,ci95_high,n" > "$MPROF"

eval_record () {  # $1=K $2=run_dir $3=csv -> éval checkpoint step 5000 (cooldown) ou 0 (merge)
  local K=$1 OUT=$2 CSV=$3 STEP=$4
  if [ -s "$OUT/checkpoints/$(printf '%06d' $STEP)/pretrained_model/model.safetensors" ]; then
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$OUT" --steps "$STEP" --n $N --kp 50 --out "$OUT/r.csv"
    local row; row=$(tail -1 "$OUT/r.csv")
    echo "${K},$(echo "$row"|cut -d, -f4),$(echo "$row"|cut -d, -f5),$(echo "$row"|cut -d, -f6),${N}" >> "$CSV"
    echo "[88] ${K}k -> $(echo "$row"|cut -d, -f4)"
  fi
  rm -rf "$OUT/checkpoints"
}

echo "[88] $(date '+%H:%M') attente Principal libre..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|12_eval_parallel|50_train_valloss|84_cooldown|33_river|90_minicnn" >/dev/null; do sleep 60; done

# --- MERGES (cheap, local) : SWA 5 ckpts centrés sur K (2k-espacés, clampé >=2k) ---
for K in $MERGE_PTS; do
  grep -q "^${K}," "$MPROF" && { echo "[88] merge ${K}k déjà fait"; continue; }
  LO=$(( K-4 < 2 ? 2 : K-4 ))
  STEPS=$(printf '%d000,%d000,%d000,%d000,%d000' $LO $((LO+2)) $((LO+4)) $((LO+6)) $((LO+8)))
  OUT=results/runs/can/joint_merge_${K}k
  $PY experiments/can/65_swa_make.py --src "$SRC_RUN" --steps "$STEPS" --out "$OUT" 2>&1 | tail -1
  eval_record "$K" "$OUT" "$MPROF" 0
done

# --- COOLDOWNS OWN (train+éval local) ---
for K in $COOLD_OWN; do
  grep -q "^${K}," "$PROF" && { echo "[88] cooldown ${K}k déjà fait"; continue; }
  S=$(printf '%06d' $((K*1000))); SRC=$SRC_RUN/checkpoints/$S/pretrained_model
  [ -s "$SRC/model.safetensors" ] || { echo "[88] source ${K}k absente"; continue; }
  CFG=/tmp/cd_config_${K}.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
  OUT=results/runs/can/joint_cd_${K}k
  echo "[88] $(date '+%H:%M') COOLDOWN(local) depuis ${K}k"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
  eval_record "$K" "$OUT" "$PROF" 5000
done

# --- COOLDOWNS mac2 : attendre le checkpoint, le tirer, évaluer ---
for K in $COOLD_MAC2; do
  grep -q "^${K}," "$PROF" && { echo "[88] cooldown ${K}k déjà fait"; continue; }
  wait_for_disk
  OUT=results/runs/can/joint_cd_${K}k
  REMOTE="results/runs/can/joint_cd_${K}k/checkpoints/005000/pretrained_model/model.safetensors"
  echo "[88] $(date '+%H:%M') attente checkpoint mac2 ${K}k..."
  t=0; until ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null; do
    t=$((t+1)); [ $t -ge 200 ] && { echo "[88] ${K}k timeout"; break; }; sleep 120; done
  ssh -o BatchMode=yes mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null || continue
  mkdir -p "$OUT/checkpoints"
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - results/runs/can/joint_cd_${K}k/checkpoints/005000/pretrained_model" | tar xzf - -C . 2>/dev/null
  eval_record "$K" "$OUT" "$PROF" 5000
  ssh -o BatchMode=yes mac2 "rm -rf ~/lerobot-experiments/results/runs/can/joint_cd_${K}k" 2>/dev/null
done

echo "[88] $(date '+%H:%M') DONE. Régénération graphes."
$PY experiments/can/85_plot_cooldown_profile.py 2>&1 | tail -1
$PY experiments/can/87_plot_joint_cooldown_4panel.py 2>&1 | tail -1
echo "[88] cooldown_profile :"; cat "$PROF"
echo "[88] merge_profile :"; cat "$MPROF"
