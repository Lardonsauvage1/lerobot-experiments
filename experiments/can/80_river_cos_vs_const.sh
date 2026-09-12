#!/bin/bash
# PROFIL RIVIÈRE : COSINE vs CONSTANT sur la MÊME tâche cartésienne (run31 vs run31-jumeau).
# Test de la théorie rivière-vallée : le CONSTANT rebondit en travers de la vallée -> fond SWA
# qui MONTE le long de la rivière ; le COSINE se pose dans la vallée (LR->0) -> fond ≈ brut, plat-haut.
# Fenêtres SWA glissantes UNIFORMES (5 ckpts, 2k-espacés) aux MÊMES positions pour les 2 runs.
# Éval cartésien = 32_eval_dense_birdview.py, n=50. Tout sur Principal (ckpts déjà locaux). En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/80_river_cos_vs_const.sh > /tmp/river_cvc.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
N=50

# fenêtres : w1 4-12k, w2 12-20k, w3 20-28k, w4 28-36k, w5 32-40k
W1="4000,6000,8000,10000,12000"
W2="12000,14000,16000,18000,20000"
W3="20000,22000,24000,26000,28000"
W4="28000,30000,32000,34000,36000"
W5="32000,34000,36000,38000,40000"

echo "[80] $(date '+%H:%M') attente Principal libre (cooldown / autre éval en cours ?)..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|47_eval_joint_ensemble|50_train_valloss" >/dev/null; do sleep 60; done
echo "[80] $(date '+%H:%M') Principal libre -> construction + éval des rivières cos vs const"

# tag -> dossier source du run
build_eval () {  # $1=tag(cos|const)  $2=src_run
  local tag=$1 src=results/runs/can/$2 i=0
  for W in "$W1" "$W2" "$W3" "$W4" "$W5"; do
    i=$((i+1))
    local out=results/runs/can/river_${tag}_w${i}
    if [ ! -s "$out/r.csv" ]; then
      $PY experiments/can/65_swa_make.py --src "$src" --steps "$W" --out "$out" 2>&1 | tail -1
      [ -s "$out/checkpoints/000000/pretrained_model/model.safetensors" ] && \
        $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$out" --steps 0 --n $N --out "$out/r.csv"
    fi
    echo "[80] ${tag} w${i} = $(cut -d, -f4 "$out/r.csv" 2>/dev/null | tail -1)"
  done
}

build_eval cos   31_proprio_birdview_r34_bigunet
build_eval const run31_constLR

echo "[80] $(date '+%H:%M') tracé du graphe comparatif"
$PY experiments/can/81_plot_river_cos_vs_const.py 2>&1 | tail -1

echo "[80] DONE — fonds rivière (n=$N) :"
for tag in cos const; do
  printf "  %-6s :" "$tag"
  for i in 1 2 3 4 5; do printf " w%s=%s" "$i" "$(cut -d, -f4 results/runs/can/river_${tag}_w${i}/r.csv 2>/dev/null | tail -1)"; done
  echo
done
