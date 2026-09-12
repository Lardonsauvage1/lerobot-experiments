#!/usr/bin/env bash
# Apple batch2 PERF (Plan A : le plus gros modèle temps-réel CPU). Arg = résolution (96 ou 128).
# Données 10 Hz (dataset apple_b2_10hz_fixed_$RES). Joint 6D absolu, 1 caméra FIXE.
# Archi + grosse : R34 + U-Net[256,512,1024] (~120-150M) — milieu, pas le 263M qui sur-apprend.
# Inférence prévue en DDIM few-step (4-5 pas) → temps-réel CPU malgré la taille (réglé au déploiement).
# horizon 16 / n_action_steps 8 : à 10 Hz = prédit 1,6 s, exécute 0,8 s (budget d'inférence).
# LR const 1e-4 + EMA + cooldown (séparé). Dataset images.
set -u
RES=$1
cd ~/lerobot-experiments
. venv/bin/activate
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
EPS=$(python -c "print(list(range(70)))" | tr -d ' ')   # train 0-69, val-loss auto 70-78
STEPS=${STEPS:-40000}
DIMS=${DIMS:-[256,512,1024]}
echo "[apple-b2-perf-h32-$RES $(date '+%H:%M')] 1cam fixe ${RES}px 10Hz, R34+$DIMS, DDIM, horizon32/nact8, const+EMA, $STEPS steps"
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 python -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=$DIMS \
  --policy.horizon=32 --policy.n_action_steps=8 --policy.n_obs_steps=2 \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=5 \
  --policy.device=cuda --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/apple_b2_10hz_fixed_$RES --dataset.root=data_cache/lerobot_apple_b2_10hz_fixed_$RES --dataset.episodes=$EPS \
  --output_dir=results/runs/real/apple_b2_perf_h32_$RES \
  --batch_size=64 --steps=$STEPS --save_freq=2000 --log_freq=50 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=8 2>&1 | tee /tmp/apple_b2_perf_h32_$RES.log
echo "[apple-b2-perf-h32-$RES $(date '+%H:%M')] TRAIN DONE"
