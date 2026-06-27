#!/bin/bash
# PROFIL RIVIÈRE mini-CNN Can : CONSTANT 1e-4 vs COSINE SGDR (restart 20/50/80/150/200/250k).
# Pour chaque run : fenêtres SWA glissantes (5 ckpts) -> fond SWA vs step, les 2 rivières superposées.
# Théorie rivière-vallée : constant rebondit -> fond qui monte ; cosine se pose dans la vallée à chaque
# vague -> fond ~ brut au creux, et on voit si CHAQUE restart se pose PLUS HAUT (progrès global SGDR).
# Éval mini-CNN = 12_eval_parallel.py (CPU, n=500). SWA = 65_swa_make.py. Tout sur Principal. En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/phase5_methodology/33_river_minicnn_cos_vs_const.sh > /tmp/river_mini.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
R=results/runs/phase5_methodology
N=500
EVAL="$PY -u experiments/phase5_methodology/12_eval_parallel.py --workers 10 --max-steps 200 --infer-steps 4 --n $N"

# --- 1) consolider les checkpoints éclatés en un seul dossier par run (symlinks, non destructif) ---
consolidate () {  # $1=dest  $2..=src run dirs
  local dest=$R/$1; shift
  mkdir -p "$dest/checkpoints"
  for src in "$@"; do
    [ -d "$R/$src/checkpoints" ] || continue
    for ck in "$R/$src/checkpoints"/*/; do
      local step=$(basename "$ck")
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      [ -e "$dest/checkpoints/$step" ] && continue
      ln -s "$(cd "$ck" && pwd)" "$dest/checkpoints/$step"
    done
  done
  echo "[33] $1 : $(ls "$dest/checkpoints" | grep -cE '^[0-9]+$') checkpoints consolidés"
}
consolidate mini_constant_all mini_constant mini_constant_continue mini_constant_continue2 mini_constant_resume mini_constant_150k
consolidate mini_cosine_all   mini_cosine   mini_cosine_continue   mini_cosine_continue2   mini_cosine_150k mini_cosine_200k mini_cosine_250k mini_cosine_300k

# --- 2) fenêtres (5 ckpts). Communes cos+const sur 0-150k ; extension cosine seule 150-250k ---
#  label   start_k end_k   steps(CSV)
COMMON=(
  "w1 4 20    4000,8000,12000,16000,20000"
  "w2 25 45   25000,30000,35000,40000,45000"
  "w3 60 80   60000,65000,70000,75000,80000"
  "w4 85 105  85000,90000,95000,100000,105000"
  "w5 110 130 110000,115000,120000,125000,130000"
  "w6 130 150 130000,135000,140000,145000,150000"
)
COS_EXT=(
  "w7 160 200 160000,170000,180000,190000,200000"
  "w8 210 250 210000,220000,230000,240000,250000"
)

# --- 2b) BALAYAGE DE TAILLE de fenêtre (levier dominant WSM : durée du merge) ---
#  À fin fixe, on élargit le merge 5 -> 10 -> 15 ckpts. Teste : moyenner PLUS de checkpoints élève-t-il le fond ?
# constant @150k (espacement 5k)
SWEEP_CONST=(
  "L5  130 150 130000,135000,140000,145000,150000"
  "L10 105 150 105000,110000,115000,120000,125000,130000,135000,140000,145000,150000"
  "L15 80 150  80000,85000,90000,95000,100000,105000,110000,115000,120000,125000,130000,135000,140000,145000,150000"
)
# cosine @150k (espacement 1k)
SWEEP_COS150=(
  "M5  146 150 146000,147000,148000,149000,150000"
  "M10 141 150 141000,142000,143000,144000,145000,146000,147000,148000,149000,150000"
  "M15 136 150 136000,137000,138000,139000,140000,141000,142000,143000,144000,145000,146000,147000,148000,149000,150000"
)
# cosine @250k (espacement 2k)
SWEEP_COS250=(
  "E5  242 250 242000,244000,246000,248000,250000"
  "E10 232 250 232000,234000,236000,238000,240000,242000,244000,246000,248000,250000"
  "E15 222 250 222000,224000,226000,228000,230000,232000,234000,236000,238000,240000,242000,244000,246000,248000,250000"
)

echo "[33] $(date '+%H:%M') attente Principal libre (cooldown / éval en cours ?)..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|47_eval_joint_ensemble|50_train_valloss|12_eval_parallel" >/dev/null; do sleep 60; done
echo "[33] $(date '+%H:%M') Principal libre -> rivières mini-CNN"

build_eval () {  # $1=tag(const|cos)  $2=src_all  shift 2 -> fenêtres "label sk ek steps"
  local tag=$1 src=$2; shift 2
  for spec in "$@"; do
    set -- $spec; local label=$1 steps=$4
    local out=$R/river_${tag}_${label}
    if [ ! -s "$out/r.csv" ]; then
      $PY experiments/can/65_swa_make.py --src "$R/$src" --steps "$steps" --out "$out" 2>&1 | tail -1
      [ -s "$out/checkpoints/000000/pretrained_model/model.safetensors" ] && \
        $EVAL --run-dir "$out" --steps 0 --out "$out/r.csv"
    fi
    echo "[33] ${tag} ${label} = $(cut -d, -f4 "$out/r.csv" 2>/dev/null | tail -1)"
  done
}

build_eval const mini_constant_all "${COMMON[@]}"
build_eval cos   mini_cosine_all   "${COMMON[@]}" "${COS_EXT[@]}"

echo "[33] $(date '+%H:%M') --- balayage de TAILLE de fenêtre (5/10/15 ckpts) ---"
build_eval const mini_constant_all "${SWEEP_CONST[@]}"
build_eval cos   mini_cosine_all   "${SWEEP_COS150[@]}" "${SWEEP_COS250[@]}"

echo "[33] $(date '+%H:%M') tracé des graphes"
$PY experiments/phase5_methodology/34_plot_river_minicnn.py 2>&1 | tail -1
$PY experiments/phase5_methodology/35_plot_windowsize_minicnn.py 2>&1 | tail -1

echo "[33] DONE — fonds rivière mini-CNN (n=$N) :"
for tag in const cos; do
  printf "  %-6s :" "$tag"
  for w in w1 w2 w3 w4 w5 w6 w7 w8; do
    v=$(cut -d, -f4 $R/river_${tag}_${w}/r.csv 2>/dev/null | tail -1)
    [ -n "$v" ] && printf " %s=%s" "$w" "$v"
  done; echo
done
echo "[33] BALAYAGE TAILLE (5/10/15 ckpts à fin fixe) :"
printf "  const @150k :"; for w in L5 L10 L15; do printf " %s=%s" "$w" "$(cut -d, -f4 $R/river_const_${w}/r.csv 2>/dev/null | tail -1)"; done; echo
printf "  cos   @150k :"; for w in M5 M10 M15; do printf " %s=%s" "$w" "$(cut -d, -f4 $R/river_cos_${w}/r.csv 2>/dev/null | tail -1)"; done; echo
printf "  cos   @250k :"; for w in E5 E10 E15; do printf " %s=%s" "$w" "$(cut -d, -f4 $R/river_cos_${w}/r.csv 2>/dev/null | tail -1)"; done; echo
