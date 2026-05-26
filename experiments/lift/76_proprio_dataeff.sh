#!/bin/bash
# Lift vision-only (9D, sans coords cube) — courbe efficacité-données.
# Entraîne N=100/50/20/10 (même arche que 73 : ResNet18 + [64,128,256], 8000 steps),
# puis lance l'éval 500 rollouts (77). N=150 = 73_proprio (déjà fait).
# Lancer : nohup bash experiments/lift/76_proprio_dataeff.sh > results/logs/lift/run_76_proprio_dataeff.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python

for N in 100 50 20 10; do
  out="results/runs/lift/73_proprio_n${N}"
  if [ -d "$out/checkpoints/008000" ]; then
    echo "[orch] $(date +%H:%M:%S) $out déjà entraîné, skip"; continue
  fi
  rm -rf "$out"
  EPS=$($PY -c "print(list(range($N)))" | tr -d ' ')
  echo "[orch] $(date +%H:%M:%S) TRAIN N=$N -> $out"
  $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.down_dims="[64,128,256]" --policy.device=mps \
    --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/lift_ph_proprio --dataset.root=data_cache/lerobot_lift_ph_proprio \
    --dataset.episodes="$EPS" \
    --output_dir="$out" \
    --batch_size=32 --steps=8000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=2 \
    || echo "[orch] $(date +%H:%M:%S) !!! TRAIN ÉCHOUÉ N=$N (on continue)"
done

echo "[orch] $(date +%H:%M:%S) entraînements finis — éval courbe 500 rollouts"
$PY -u experiments/lift/77_proprio_dataeff_eval.py || echo "[orch] $(date +%H:%M:%S) !!! ÉVAL ÉCHOUÉE"
echo "[orch] $(date +%H:%M:%S) PROPRIO DATAEFF DONE"
