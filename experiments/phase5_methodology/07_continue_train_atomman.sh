#!/bin/bash
# Phase 5 — Continue train sur atomman CPU, 20k steps, robuste aux reboots.
#
# Deux modes automatiques :
#   - RESUME : si OUT contient déjà un checkpoint (checkpoints/last), on reprend
#     exactement où on en était (step + optimizer + lr_scheduler via --resume=true).
#     => les reboots d'atomman ne perdent plus que <500 steps (depuis le dernier save).
#   - FRESH  : sinon, démarrage à neuf depuis les poids du ckpt 30k (optimizer frais).
#     Un dir OUT incomplet (sans checkpoint exploitable) est sauvegardé de côté avant.
#
# Lancer : nohup bash experiments/phase5_methodology/07_continue_train_atomman.sh > /tmp/run_07_continue.log 2>&1 &

cd "/home/sam/Documents/projet VScode/experience_Le" || exit 1
PY=venv312/bin/python
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12   # P-cores only (test accel 2026-06-11 : 11.2 vs 16.9 s/step)

CKPT_30K_WEIGHTS="results/runs/phase5_methodology/26_resnet34_dense/checkpoints/030000/pretrained_model"
OUT="results/runs/phase5_methodology/26_resnet34_dense_continue"
LAST_CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"

# ---------------------------------------------------------------------------
# MODE RESUME : checkpoint exploitable présent -> reprise exacte
# ---------------------------------------------------------------------------
if [ -f "$LAST_CFG" ]; then
  STEP=$(basename "$(readlink -f "$OUT/checkpoints/last")")
  echo "[07] $(date '+%H:%M:%S') RESUME depuis checkpoint $STEP -> $OUT"
  taskset -c 0-11 $PY -u experiments/lift/50_train_valloss.py \
    --config_path="$LAST_CFG" \
    --resume=true \
    || { echo "[07] !!! RESUME ÉCHOUÉ" ; exit 1 ; }
  echo "[07] $(date '+%H:%M:%S') CONTINUE TRAIN DONE"
  exit 0
fi

# ---------------------------------------------------------------------------
# MODE FRESH : pas de checkpoint exploitable -> démarrage à neuf depuis 30k
# ---------------------------------------------------------------------------
# Si un dir OUT existe mais sans checkpoint utilisable, on le met de côté
# (sinon lerobot lèverait FileExistsError).
if [ -d "$OUT" ]; then
  BK="${OUT}_stale_$(date '+%Y%m%d_%H%M%S')"
  mv "$OUT" "$BK" && echo "[07] dir OUT incomplet sauvegardé -> $BK"
fi

# Patcher la config du ckpt source en place (device mps -> cpu)
$PY -c "
import json
from pathlib import Path
p = Path('$CKPT_30K_WEIGHTS/train_config.json')
cfg = json.loads(p.read_text())
cfg['policy']['device'] = 'cpu'
p.write_text(json.dumps(cfg, indent=2))
for fn in ['policy_preprocessor.json', 'policy_postprocessor.json']:
    pp = Path('$CKPT_30K_WEIGHTS') / fn
    if pp.exists():
        c = json.loads(pp.read_text())
        for step in c.get('steps', []):
            if step.get('registry_name') == 'device_processor':
                step['config']['device'] = 'cpu'
        pp.write_text(json.dumps(c, indent=2))
print('configs patched -> cpu')
"

EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')

echo "[07] $(date '+%H:%M:%S') FRESH continue 20k steps from ckpt 30k weights -> $OUT"
taskset -c 0-11 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion \
  --policy.down_dims="[64,128,256]" \
  --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.pretrained_path="$CKPT_30K_WEIGHTS" \
  --policy.device=cpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_birdview \
  --dataset.root=data_cache/lerobot_can_ph_proprio_birdview \
  --dataset.episodes="$EPS" \
  --output_dir="$OUT" \
  --batch_size=32 --steps=20000 --save_freq=500 --log_freq=100 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=0 \
  || { echo "[07] !!! FRESH CONTINUE TRAIN ÉCHOUÉ" ; exit 1 ; }
echo "[07] $(date '+%H:%M:%S') CONTINUE TRAIN DONE"
