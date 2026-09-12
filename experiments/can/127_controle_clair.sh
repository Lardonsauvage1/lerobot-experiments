#!/usr/bin/env bash
# LE CONTRÔLE QUI MANQUAIT DEPUIS LE DÉBUT.
#
# Trois fois de suite j'ai cherché le coupable par élimination (banc, dataset, architecture)
# sans jamais poser le point de référence qui rend chaque test interprétable :
#   « ma configuration atteint-elle le niveau attendu sur des données SAINES ? »
#
# Ici on ne change QU'UNE chose par rapport au bras A v2 : le dataset (clair au lieu
# d'occlus). Tout le reste est identique — ResNet34 + [64,128,256], batch 32, 10k steps.
#
#   ~70 %  -> le pipeline est sain ; c'est l'occlusion à 3 cm qui est trop dure à apprendre
#             avec ce budget (référence : run 02, mono-caméra, 320k échantillons -> 70,4 % @500)
#   ~4 %   -> le problème est dans ma config ; suspect n°1 = ResNet34 (le run 02 utilise
#             ResNet18, seule différence structurelle restante)
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/127_controle.log
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
step() { echo "" >> $LOG; echo "[$(date '+%m-%d %H:%M:%S')] ═══ $* ═══" >> $LOG; }

step "ATTENTE : fin de l'éval baseline v2, puis arrêt AVANT le bras B (inexploitable si la baseline est à 3 %)"
while pgrep -f "126_chaine_v2" >/dev/null; do
  [ -f results/runs/can/occluded/eval_v2_baseline.json ] && break
  sleep 30
done
pkill -f "126_chaine_v2"; sleep 3; pkill -f "50_train_valloss"
echo "[ctrl] chaîne v2 stoppée avant le bras B" >> $LOG

step "ENTRAÎNEMENT sur dataset CLAIR — config identique au bras A v2"
rm -rf results/runs/can/ctrl_clair
EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion --policy.vision_backbone=resnet34 "--policy.down_dims=[64,128,256]" \
  --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio --dataset.root=data_cache/lerobot_can_ph_proprio_img \
  --dataset.episodes="$EPS" --output_dir=results/runs/can/ctrl_clair \
  --batch_size=32 --steps=10000 --save_freq=5000 --log_freq=500 --eval_freq=0 \
  --seed=42 --wandb.enable=false --num_workers=2 >> $LOG 2>&1

step "COOLDOWN"
SRC=results/runs/can/ctrl_clair/checkpoints/010000/pretrained_model
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('/tmp/cd_ctrl.json','w'))"
EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=/tmp/cd_ctrl.json --resume=false --policy.pretrained_path="$SRC" \
  --output_dir=results/runs/can/ctrl_clair/cooldown --steps=5000 --save_freq=5000 \
  --log_freq=500 --eval_freq=0 >> $LOG 2>&1

step "ÉVAL SANS occlusion, 150 rollouts (comparable au run 02 : 70,4 % @500)"
$PY -u experiments/can/120_eval_occluded.py \
  --ckpt results/runs/can/ctrl_clair/cooldown/checkpoints/005000/pretrained_model \
  --radius 0.0 --n 150 --tag ctrl_clair >> $LOG 2>&1

step "CONTRÔLE TERMINÉ"
