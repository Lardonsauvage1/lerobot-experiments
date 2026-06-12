#!/bin/bash
# Phase 5 — Variance FINE : densifier une fenetre de 200 pas JAMAIS entrainee (option B).
# Prolonge le vrai run constant au-dela de 80k, 1 checkpoint tous les 10 pas (premier passage,
# pas de re-train d'une zone deja vue -> pas d'ambiguite de trajectoire).
#
# A lancer sur ATOMMAN une fois que mini_constant_continue2 a produit le ckpt 080000.
#   ssh atomman ... bash experiments/phase5_methodology/16_finevar_dense.sh
set -e
cd "/home/sam/Documents/projet VScode/experience_Le"
ROOT="results/runs/phase5_methodology"
INIT="$ROOT/mini_constant_continue2/checkpoints/080000/pretrained_model"
OUT="$ROOT/mini_constant_finevar"
[ -f "$INIT/model.safetensors" ] || { echo "ckpt 80000 absent: $INIT"; exit 1; }

export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12

# 1) train 80000 -> 80200, save tous les 10 (-> 80010..80200)
echo "[16] $(date '+%H:%M:%S') train dense 80000->80200 (save/10)"
taskset -c 0-11 bash experiments/phase5_methodology/14_continue_minicnn.sh \
  constant cpu 80200 10 "$INIT" 80000 "$OUT"

# 2) inclure l'ancre 80000 dans le dir finevar (pour l'avoir dans la courbe)
mkdir -p "$OUT/checkpoints/080000"
cp -r "$INIT" "$OUT/checkpoints/080000/pretrained_model"

# 3) eval 500-rollouts des 21 checkpoints (80000, 80010, ..., 80200)
S=$(seq 80000 10 80200 | tr '\n' ',' | sed 's/,$//')
echo "[16] $(date '+%H:%M:%S') eval 500 des 21 ckpts: $S"
unset MUJOCO_GL
venv312/bin/python -u experiments/phase5_methodology/12_eval_parallel.py \
  --run-dir "$OUT" --steps "$S" --n 500 --workers 12 --max-steps 200 --infer-steps 4 \
  --out "$OUT/rollouts_500.csv"
echo "[16] $(date '+%H:%M:%S') DONE finevar"
