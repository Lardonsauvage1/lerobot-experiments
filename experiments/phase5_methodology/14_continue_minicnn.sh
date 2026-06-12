#!/bin/bash
# Phase 5 — CONTINUE mini-CNN 20k -> 50k (Can, vision pure).
#
# Le resume natif LeRobot est CASSÉ pour le mini-CNN (le swap recree des poids
# aleatoires apres le load). On contourne via 3 hooks env dans 61_train_minicnn.py :
#   MINI_INIT_CKPT  : recharge les poids du ckpt 20k par-dessus le mini-CNN (strict=False)
#   MINI_SCHED      : scheduler custom pour 20k->50k  (cosine_wave = 1 vague SGDR | constant)
#   MINI_START_STEP : reprend la numerotation des checkpoints a 20000 (-> 021000..050000)
#
# Usage :  bash 14_continue_minicnn.sh <cosine_wave|constant> <mps|cpu> [STEPS] [SAVE_FREQ] [INIT_CKPT] [START_STEP] [OUT_DIR]
#   defauts (phase 20k->50k) : INIT=<src>/checkpoints/020000, START=20000, OUT=<src>_continue
#   exemple phase 50k->80k   : ... 80000 1000 <...>/050000/pretrained_model 50000 <...>_continue2
set -e
MODE="${1:?mode = cosine_wave | constant}"
DEV="${2:?device = mps | cpu}"
STEPS="${3:-50000}"
SAVE="${4:-1000}"
INIT_OVERRIDE="${5:-}"
START="${6:-20000}"
OUT_OVERRIDE="${7:-}"

case "$MODE" in
  cosine_wave) SRC=mini_cosine ;;
  constant)    SRC=mini_constant ;;
  *) echo "mode inconnu: $MODE"; exit 1 ;;
esac

ROOT="results/runs/phase5_methodology"
INIT="${INIT_OVERRIDE:-$ROOT/$SRC/checkpoints/020000/pretrained_model}"
OUT="${OUT_OVERRIDE:-$ROOT/${SRC}_continue}"
[ -f "$INIT/model.safetensors" ] || { echo "ckpt de depart absent: $INIT"; exit 1; }

EPS="$(venv312/bin/python -c 'print(list(range(150)))' | tr -d ' ')"

export MINI_INIT_CKPT="$INIT"
export MINI_SCHED="$MODE"
export MINI_START_STEP="$START"
export MINI_WARMUP=500

echo "[14] $(date '+%H:%M:%S') CONTINUE $MODE  $SRC -> $OUT  (device=$DEV steps=$STEPS save=$SAVE)"
venv312/bin/python -u experiments/lift/61_train_minicnn.py \
  --policy.type=diffusion \
  --policy.down_dims="[32,64,128]" \
  --policy.vision_backbone=resnet18 \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.crop_shape=null --policy.push_to_hub=false --policy.device="$DEV" \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir="$OUT" \
  --batch_size=32 --steps="$STEPS" --save_freq="$SAVE" --log_freq=100 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=0
echo "[14] $(date '+%H:%M:%S') DONE $MODE"
