#!/bin/bash
# Run 75 sur atomman (CPU 22 threads) : ResNet18 + U-Net [256,512,1024] vision pure 9D.
# 15 000 steps, save_freq 3000 (5 checkpoints), batch 32.
# Puis éval 500 rollouts immédiate avec 82_run75_eval_500.py.
# Lancer : nohup bash experiments/lift/83_run75_train_eval.sh > results/logs/lift/run_83_run75.log 2>&1 &

cd "/home/sam/Documents/projet VScode/experience_Le" || exit 1
PY=venv312/bin/python
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22

EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
OUT="results/runs/lift/75_proprio_big"

if [ -d "$OUT/checkpoints" ]; then
  echo "[83] $(date '+%H:%M:%S') $OUT déjà entraîné, eval direct"
else
  rm -rf "$OUT"
  echo "[83] $(date '+%H:%M:%S') TRAIN run 75 (ResNet18 + [256,512,1024]) -> $OUT"
  $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion \
    --policy.down_dims="[256,512,1024]" \
    --policy.device=cpu --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/lift_ph_proprio --dataset.root=data_cache/lerobot_lift_ph_proprio \
    --dataset.episodes="$EPS" \
    --output_dir="$OUT" \
    --batch_size=32 --steps=15000 --save_freq=3000 --log_freq=100 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=0 \
    || { echo "[83] $(date '+%H:%M:%S') !!! TRAIN ÉCHOUÉ" ; exit 1 ; }
fi

echo "[83] $(date '+%H:%M:%S') EVAL 500 rollouts"
$PY -u experiments/lift/82_run75_eval_500.py --run-dir "$OUT" \
  || echo "[83] $(date '+%H:%M:%S') !!! EVAL ÉCHOUÉE"
echo "[83] $(date '+%H:%M:%S') RUN 75 COMPLETE"
