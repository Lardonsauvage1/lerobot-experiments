#!/bin/bash
# PROFIL COOLDOWN (parallélisé Principal + mac2). 7 sources : cooldown 5k (LR 1e-4->0),
# garde SEULEMENT le checkpoint LR=0, éval n=100, supprime. But : annealing réel vs merge (rivière).
#   Principal : entraîne+évalue OWN={30,50,70,80} ; puis TIRE de mac2 et évalue MAC2={20,40,60}.
#   (mac2 entraîne MAC2 via 86_cooldown_mac2.sh ; seules les évals tournent sur Principal = robosuite.)
# Reprenable (skippe ce qui est déjà dans le CSV). En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/84_cooldown_sweep.sh > /tmp/cd_sweep.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
SRC_RUN=results/runs/can/joint_r34_bigunet
PROF=results/runs/can/cooldown_profile.csv
N=100
OWN="70"; MAC2="20 40 60 80"
[ -s "$PROF" ] || echo "source_k,success_rate,ci95_low,ci95_high,n" > "$PROF"

eval_record () {  # $1=K  $2=run_dir  -> éval le checkpoint LR=0 (step 5000), écrit le CSV, nettoie
  local K=$1 OUT=$2
  if [ -s "$OUT/checkpoints/005000/pretrained_model/model.safetensors" ]; then
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$OUT" --steps 5000 --n $N --kp 50 --out "$OUT/r.csv"
    local row; row=$(tail -1 "$OUT/r.csv")
    echo "${K},$(echo "$row"|cut -d, -f4),$(echo "$row"|cut -d, -f5),$(echo "$row"|cut -d, -f6),${N}" >> "$PROF"
    echo "[84] ${K}k -> LR=0 = $(echo "$row"|cut -d, -f4)"
  else
    echo "[84] ${K}k : checkpoint LR=0 absent"
  fi
  rm -rf "$OUT/checkpoints"
}

echo "[84] $(date '+%H:%M') attente Principal libre (rivière mini-CNN)..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|12_eval_parallel|50_train_valloss|33_river_minicnn" >/dev/null; do sleep 60; done

# --- Phase A : sources OWN, entraînées + évaluées localement ---
for K in $OWN; do
  grep -q "^${K}," "$PROF" && { echo "[84] ${K}k déjà fait"; continue; }
  S=$(printf '%06d' $((K*1000))); SRC=$SRC_RUN/checkpoints/$S/pretrained_model
  [ -s "$SRC/model.safetensors" ] || { echo "[84] source ${K}k absente"; continue; }
  CFG=/tmp/cd_config_${K}.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
  OUT=results/runs/can/joint_cd_${K}k
  echo "[84] $(date '+%H:%M') COOLDOWN(local) depuis ${K}k"
  COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
    --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
    --output_dir=$OUT --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
  eval_record "$K" "$OUT"
done

# --- Phase B : sources MAC2, entraînées sur mac2 -> on tire le checkpoint et on évalue ---
for K in $MAC2; do
  grep -q "^${K}," "$PROF" && { echo "[84] ${K}k déjà fait"; continue; }
  OUT=results/runs/can/joint_cd_${K}k
  REMOTE="results/runs/can/joint_cd_${K}k/checkpoints/005000/pretrained_model/model.safetensors"
  echo "[84] $(date '+%H:%M') attente du checkpoint mac2 ${K}k (max ~5h)..."
  tries=0
  until ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null; do
    tries=$((tries+1)); [ $tries -ge 150 ] && { echo "[84] ${K}k timeout mac2"; break; }
    sleep 120
  done
  ssh -o BatchMode=yes mac2 "[ -s ~/lerobot-experiments/$REMOTE ]" 2>/dev/null || continue
  echo "[84] $(date '+%H:%M') tire ${K}k depuis mac2"
  mkdir -p "$OUT/checkpoints"
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - results/runs/can/joint_cd_${K}k/checkpoints/005000/pretrained_model" | tar xzf - -C . 2>/dev/null
  eval_record "$K" "$OUT"
  ssh -o BatchMode=yes mac2 "rm -rf ~/lerobot-experiments/results/runs/can/joint_cd_${K}k" 2>/dev/null
done

echo "[84] $(date '+%H:%M') DONE — profil cooldown :"
cat "$PROF"
$PY experiments/can/85_plot_cooldown_profile.py 2>&1 | tail -1
