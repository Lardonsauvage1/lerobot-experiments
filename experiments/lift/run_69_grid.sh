#!/bin/bash
# Orchestrateur grille 15 (Phase 4) : attend 67 -> entraîne les 10 modèles mini-CNN
# manquants ([64,128,256] et [128,256,512] × N=150/100/50/20/10) -> éval 500 rollouts.
# Recette d'entraînement IDENTIQUE aux modèles dataeff (via --config_path + overrides).
# Lancer détaché : nohup bash experiments/lift/run_69_grid.sh > results/logs/lift/run_69_grid.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
CFG=results/runs/lift/63_dataeff_n100/checkpoints/006000/pretrained_model/train_config.json
DIMS=("[64,128,256]" "[128,256,512]")
NS=(150 100 50 20 10)

echo "[orch] $(date '+%H:%M:%S') waiting for 67_dataeff_500 to finish (frees GPU + phase4_eval500.npy)..."
while pgrep -f 67_dataeff_500.py >/dev/null 2>&1; do sleep 30; done
echo "[orch] $(date '+%H:%M:%S') 67 done — starting trainings."

for d in "${DIMS[@]}"; do
  dtag=$(echo "$d" | tr -d '[] ' | tr ',' '_')
  for N in "${NS[@]}"; do
    out="results/runs/lift/69_grid_d${dtag}_n${N}"
    if [ -d "$out/checkpoints/006000" ]; then
      echo "[orch] $(date '+%H:%M:%S') $out déjà entraîné, skip"; continue
    fi
    rm -rf "$out"  # nettoie un éventuel run partiel (resume=False)
    eps=$($PY -c "print(list(range($N)))" | tr -d ' ')
    echo "[orch] $(date '+%H:%M:%S') TRAIN d=$d N=$N -> $out"
    $PY -u experiments/lift/61_train_minicnn.py \
      --config_path="$CFG" \
      --policy.down_dims="$d" \
      --dataset.episodes="$eps" \
      --output_dir="$out" \
      --steps=6000 --save_freq=750 --log_freq=50 --eval_freq=0 \
      || echo "[orch] $(date '+%H:%M:%S') !!! TRAIN ÉCHOUÉ d=$d N=$N (on continue)"
  done
done

echo "[orch] $(date '+%H:%M:%S') tous les entraînements finis — éval grille 500 rollouts."
$PY -u experiments/lift/69_grid_eval.py \
  || echo "[orch] $(date '+%H:%M:%S') !!! ÉVAL GRILLE ÉCHOUÉE"
echo "[orch] $(date '+%H:%M:%S') GRID COMPLETE"
