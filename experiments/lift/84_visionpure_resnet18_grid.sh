#!/bin/bash
# Grille ResNet18 vision pure (complète phase 4 mais sans coords cube).
# 2 U-Nets nouveaux × 5 N - 1 déjà fait (run 75) = 9 cellules à entraîner.
# Pour [256,512,1024] × N=150 : déjà fait par run 75 (75_proprio_big), skip.
# Pour [64,128,256] × tous N : déjà fait par run 73 + 76 (proprio_dataeff_500.json), skip.
# Chaîné : train + eval 500 rollouts par cellule. Cellule = 1 train + 1 eval.
# Lancer : nohup bash experiments/lift/84_visionpure_resnet18_grid.sh > results/logs/lift/run_84_visionpure_resnet18.log 2>&1 &

cd "/home/sam/Documents/projet VScode/experience_Le" || exit 1
PY=venv312/bin/python
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22

DIMS=("[128,256,512]" "[256,512,1024]")
NS=(150 100 50 20 10)
STEPS=15000

for d in "${DIMS[@]}"; do
  dtag=$(echo "$d" | tr -d '[] ' | tr ',' '_')
  for N in "${NS[@]}"; do
    out="results/runs/lift_visionpure_resnet/d${dtag}_n${N}"
    eval_json="results/runs/lift_visionpure_resnet/d${dtag}_n${N}_eval500.json"
    if [ -f "$eval_json" ]; then
      echo "[orch] $(date '+%H:%M:%S') $out déjà évalué, skip" ; continue
    fi
    # Skip explicite si déjà fait sous autre nom
    if [ "$d" = "[256,512,1024]" ] && [ "$N" = "150" ]; then
      echo "[orch] $(date '+%H:%M:%S') [256,512,1024] × N=150 = run 75 (75_proprio_big), skip" ; continue
    fi
    if [ ! -d "$out/checkpoints" ]; then
      rm -rf "$out"
      EPS=$($PY -c "print(list(range($N)))" | tr -d ' ')
      echo "[orch] $(date '+%H:%M:%S') TRAIN d=$d N=$N -> $out"
      $PY -u experiments/lift/50_train_valloss.py \
        --policy.type=diffusion --policy.down_dims="$d" \
        --policy.device=cpu --policy.push_to_hub=false --policy.crop_shape=null \
        --dataset.repo_id=local/lift_ph_proprio --dataset.root=data_cache/lerobot_lift_ph_proprio \
        --dataset.episodes="$EPS" \
        --output_dir="$out" \
        --batch_size=32 --steps=$STEPS --save_freq=3000 --log_freq=100 --eval_freq=0 \
        --seed=42 --wandb.enable=false --num_workers=0 \
        || { echo "[orch] $(date '+%H:%M:%S') !!! TRAIN ÉCHOUÉ d=$d N=$N (skip eval)" ; continue ; }
    else
      echo "[orch] $(date '+%H:%M:%S') $out déjà entraîné, eval direct"
    fi
    echo "[orch] $(date '+%H:%M:%S') EVAL 500 rollouts d=$d N=$N"
    $PY -u experiments/lift/82_run75_eval_500.py --run-dir "$out" \
      || echo "[orch] $(date '+%H:%M:%S') !!! EVAL ÉCHOUÉE d=$d N=$N"
  done
done

echo "[orch] $(date '+%H:%M:%S') GRID RESNET18 COMPLETE"
