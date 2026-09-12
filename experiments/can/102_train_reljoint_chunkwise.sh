#!/bin/bash
# Delta CHUNK-WISE (joint RELATIF, ancre = obs courante) — LA bonne variante (cf. Demystifying Action Space).
# Réutilise le dataset ABSOLU ; le relatif se calcule à la volée (RELJOINT=1 patch 50_train_valloss).
# From scratch, CONST_LR 1e-4, 40k. Puis cooldown 5k + éval chunk-wise (44_eval_reljoint) -> VERDICT vs absolu 83%.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
RUN=results/runs/can/joint_r34_reljoint_cw
mkdir -p results/logs/can

echo "[102] $(date '+%H:%M') TRAIN delta chunk-wise (RELJOINT=1) from scratch, 40k, CONST_LR 1e-4"
RELJOINT=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims="[128,256,512]" --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_joint_birdview --dataset.root=data_cache/lerobot_can_ph_joint_birdview --dataset.episodes="$EPS" \
  --output_dir=$RUN \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=500 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 \
  2>&1 | tee results/logs/can/run_reljoint_cw.log
echo "[102] $(date '+%H:%M') TRAIN DONE. Cooldown 40k + eval chunk-wise."

# --- COOLDOWN 40k (RELJOINT=1 aussi, mêmes stats relatives) + EVAL chunk-wise (règle cooldown par défaut) ---
SRC=$RUN/checkpoints/040000/pretrained_model
[ -s "$SRC/model.safetensors" ] || { echo "[102] ckpt 40k absent"; exit 1; }
CFG=/tmp/reljoint_cd.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
echo "[102] $(date '+%H:%M') COOLDOWN 40k (LR 1e-4->0 / 5k)"
RELJOINT=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown_40k --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
$PY -u experiments/can/44_eval_reljoint.py --run-dir "$RUN/cooldown_40k" --steps 5000 --n 50 --kp 50 --out "$RUN/cooldown_40k/r.csv"
echo "[102] $(date '+%H:%M') ===== VERDICT DELTA CHUNK-WISE 40k (cooldown) : $(tail -1 "$RUN/cooldown_40k/r.csv"|cut -d, -f4) (vs absolu 83%, séquentiel 0%) ====="
