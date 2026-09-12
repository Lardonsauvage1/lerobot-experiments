#!/usr/bin/env bash
# BRAS CAMP avec CANARI — config du run 02, sur le dataset occlus 3 cm.
#
# Pourquoi un canari. Le 2026-09-10, quatre hypothèses fausses ont coûté ~20 h parce que
# l'erreur n'était détectée qu'APRÈS 4-6 h d'entraînement. La parade n'est pas de mieux
# deviner, c'est de DÉTECTER TÔT.
#
# Indicateur : « la canette décolle-t-elle ? » (max z > 0,88 m), et NON le succès binaire —
# un modèle en cours d'apprentissage soulève l'objet bien avant de réussir la tâche, donc
# cet indicateur bouge des milliers de steps plus tôt.
#
# ⚠️ Cette config (ResNet18 + cosine par défaut + batch 32 + 10k) a donné 70,4 % @500…
# mais le 29 MAI, sur un dataset ANTÉRIEUR (l'actuel date du 16 juillet). Elle n'est donc
# PAS prouvée sur ces données — d'où le canari.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/129_camp.log
OUT=results/runs/can/b3_camp
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
step() { echo "" >> $LOG; echo "[$(date '+%H:%M:%S')] ═══ $* ═══" >> $LOG; }

step "ENTRAÎNEMENT baseline — ResNet18 + cosine (défaut) + batch 32 + 10k, dataset occlus 3 cm"
rm -rf "$OUT"
# NI CONST_LR NI EMA : identique à la baseline. SEUL ÉCART : CAMP=1 (+32 dims de mémoire).
export CAMP=1 CAMP_CKPT=results/runs/can/camp/memory_L64_K32_m32.pt
$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion "--policy.down_dims=[64,128,256]" \
  --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio_occ03 --dataset.root=data_cache/lerobot_can_ph_proprio_occ03 \
  --dataset.episodes="$EPS" --output_dir="$OUT" \
  --batch_size=32 --steps=10000 --save_freq=2500 --log_freq=500 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 >> $LOG 2>&1 &
TRAIN_PID=$!
echo "[canari] entraînement PID $TRAIN_PID" >> $LOG

# --- CANARI : évalue les checkpoints précoces et COUPE si le modèle n'apprend rien -------
for CK in 002500 005000; do
  while [ ! -s "$OUT/checkpoints/$CK/pretrained_model/model.safetensors" ]; do
    kill -0 $TRAIN_PID 2>/dev/null || break
    sleep 30
  done
  kill -0 $TRAIN_PID 2>/dev/null || break
  sleep 10
  step "CANARI à $CK steps — 30 rollouts SANS occlusion (le plus sensible)"
  $PY -u experiments/can/120_eval_occluded.py --ckpt "$OUT/checkpoints/$CK/pretrained_model" \
      --radius 0.0 --n 30 --camp results/runs/can/camp/memory_L64_K32_m32.pt --tag campcanari_$CK >> $LOG 2>&1
  LIFT=$($PY -c "
import json
try: print(json.load(open('results/runs/can/occluded/eval_campcanari_$CK.json'))['can_lifted_frac'])
except Exception: print(0.0)")
  echo "[canari $CK] canette soulevée : $LIFT" >> $LOG
  if $PY -c "import sys; sys.exit(0 if float('$LIFT') < 0.05 else 1)"; then
    if [ "$CK" = "005000" ]; then
      step "❌ CANARI MUET à 5000 steps (canette jamais soulevée) -> ARRÊT, la config n'apprend pas"
      kill $TRAIN_PID 2>/dev/null
      exit 2
    else
      echo "[canari $CK] rien encore, on laisse une chance jusqu'à 5000" >> $LOG
    fi
  else
    step "✅ CANARI POSITIF à $CK ($LIFT soulevé) — la config apprend, on continue"
  fi
done

wait $TRAIN_PID
step "COOLDOWN"
SRC=$OUT/checkpoints/010000/pretrained_model
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('/tmp/cd_b3.json','w'))"
CAMP=1 CAMP_CKPT=results/runs/can/camp/memory_L64_K32_m32.pt \
COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=/tmp/cd_b3.json --resume=false --policy.pretrained_path="$SRC" \
  --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0 >> $LOG 2>&1

step "ÉVAL 150 rollouts @ 3 cm"
$PY -u experiments/can/120_eval_occluded.py --ckpt $OUT/cooldown/checkpoints/005000/pretrained_model \
    --radius 0.03 --n 150 --camp results/runs/can/camp/memory_L64_K32_m32.pt --tag b3_camp >> $LOG 2>&1
step "BASELINE TERMINÉE"
