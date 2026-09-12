#!/bin/bash
# Delta COHÉRENT relancé sur PRINCIPAL (mac2 tombé) — from scratch, CONST_LR 1e-4, 40k, logs PERSISTANTS.
# Puis COOLDOWN 5k + éval (règle par défaut). Donne le VERDICT delta sans mac2.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
RUN=results/runs/can/joint_r34_delta_coh_P
mkdir -p results/logs/can

echo "[101] $(date '+%H:%M') TRAIN delta cohérent from scratch sur Principal (40k, CONST_LR 1e-4)"
CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.down_dims="[128,256,512]" --policy.vision_backbone="resnet34" \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_joint_delta_coh_birdview \
  --dataset.root=data_cache/lerobot_can_ph_joint_delta_coh_birdview --dataset.episodes="$EPS" \
  --output_dir=$RUN \
  --batch_size=16 --steps=40000 --save_freq=2000 --log_freq=500 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 \
  2>&1 | tee results/logs/can/run_delta_coh_P.log
echo "[101] $(date '+%H:%M') TRAIN DONE. Cooldown 40k + eval."

# --- COOLDOWN 40k + EVAL (règle par défaut) ---
SRC=$RUN/checkpoints/040000/pretrained_model
[ -s "$SRC/model.safetensors" ] || { echo "[101] ckpt 40k absent, stop"; exit 1; }
CFG=/tmp/dc_P_cd.json
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.get('wandb',{}).pop('add_tags',None);c.setdefault('wandb',{})['enable']=False;json.dump(c,open('$CFG','w'))"
echo "[101] $(date '+%H:%M') COOLDOWN 40k (LR 1e-4->0 / 5k)"
COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$RUN/cooldown_40k --steps=5000 --save_freq=5000 --log_freq=100 --eval_freq=0
$PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RUN/cooldown_40k" --steps 5000 --n 50 --kp 50 --delta --out "$RUN/cooldown_40k/r.csv"
echo "[101] $(date '+%H:%M') ===== VERDICT DELTA COHÉRENT 40k (cooldown) : $(tail -1 "$RUN/cooldown_40k/r.csv"|cut -d, -f4) (séquentiel/10k=0%) ====="
