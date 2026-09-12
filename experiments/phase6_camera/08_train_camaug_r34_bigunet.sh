#!/bin/bash
# Phase 6 caméra — le gros modèle 61M (= run 31 : ResNet34 + gros U-Net [128,256,512],
# encodeurs séparés, birdview) entraîné AVEC augmentation caméra (jitter par épisode ≤10cm/10°).
# But : casser le plafond de robustesse camshift (~18-20% des modèles caméra plus petits).
#
# LR CONSTANT (décision méthodo : pas de cosine -> on s'arrête au plateau, pas d'inconnue d'horizon).
#   -> CONST_LR=1 désactive le scheduler dans 50_train_valloss.py (LR fixe = CONST_LR_VALUE).
# batch 16 (falaise mémoire 16 Go, comme run 31). Mêmes checkpoints que run 31 : save_freq=2000, steps=40000.
# Dataset : camaug existant (≤10cm/10°, jitter par épisode). Checkpoints conservés.
# Lancer : nohup bash experiments/phase6_camera/08_train_camaug_r34_bigunet.sh > results/logs/phase6_camera/run_08_camaug_r34_bigunet.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
mkdir -p results/logs/phase6_camera

CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[128,256,512]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview_camaug \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview_camaug \
  --dataset.episodes="$EPS" \
  --output_dir=results/runs/phase6_camera/camaug_r34_bigunet \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2
echo "[08_camaug_r34_bigunet] TRAIN DONE"
