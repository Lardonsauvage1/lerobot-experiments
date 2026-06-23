#!/bin/bash
# Phase 5 Can — cumul des deux axes de capacité qui ont payé séparément :
#   - vision : ResNet34 (run 26 = 81.4% @500)
#   - décodeur : gros U-Net [128,256,512] (run 25 = 82% @50, jamais mesuré @500)
# = ResNet34 birdview + U-Net [128,256,512]. Le reste identique à 16/26 (birdview, 9D vision pure).
#
# BATCH 16 (pas 32) : c'est le plus gros modèle (R34 ~42M + gros U-Net ~17M). À batch 32 il
# dépasserait les 16 Go du M1 -> falaise SSD (cf docs/PERF_TEMPS_ENTRAINEMENT.md, run 25 pic 104s).
# Batch 16 halve les activations -> reste sous la limite.
# steps=40000 pour voir ~640k échantillons (= budget du R34 batch32/20k qui a fait 81.4%).
# Checkpoints conservés (save_freq=2000) — NE PAS purger (run 25 perdu faute de ckpt).
# Lancer : nohup bash experiments/can/31_train_proprio_birdview_r34_bigunet.sh > results/logs/can/run_31_r34_bigunet.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/can/31_proprio_birdview_r34_bigunet \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[31_proprio_birdview_r34_bigunet] TRAIN DONE"
