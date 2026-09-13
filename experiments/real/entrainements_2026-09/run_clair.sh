#!/usr/bin/env bash
# PLAFOND SANS OCCLUSION — le run qui manque pour chiffrer le coût réel de l occlusion.
# Config IDENTIQUE à b3_baseline (ResNet18 + [64,128,256] + cosine + batch 32 + 10k),
# seule différence : le dataset est CLAIR, régénéré avec le MÊME convertisseur que
# l occlus (même format vidéo) -> seule l occlusion varie.
#   clair/clair   : ?        <- ce run
#   occlus/occlus : 56,0 %
#   -> la différence est le coût VRAI de l occlusion, donc le maximum qu une mémoire vise.
set -u
cd ~/lerobot-experiments
PY=~/ipex_test_venv/bin/python
EPS=$($PY -c "print(list(range(150)))" | tr -d " ")
OUT=results/runs/can/clair_ref
while pgrep -f "run_small.sh" > /dev/null; do sleep 120; done
# ATTENTE DU CONTENU, pas du dossier : rsync cree l arborescence des le debut du
# transfert. Le 11/09 le run a demarre sur 86 episodes sur 200 et a echoue.
while true; do
  n=$($PY -c "import json;print(json.load(open('data_cache/lerobot_can_ph_proprio/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
  [ "$n" = "200" ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] ===== PLAFOND CLAIR ====="
rm -rf $OUT
$PY -u experiments/lift/50_train_valloss.py \
  --policy.type=diffusion "--policy.down_dims=[64,128,256]" \
  --policy.device=xpu --policy.push_to_hub=false --policy.crop_shape=null \
  --dataset.repo_id=local/can_ph_proprio \
  --dataset.root=data_cache/lerobot_can_ph_proprio --dataset.episodes="$EPS" \
  --output_dir=$OUT --batch_size=32 --steps=10000 --save_freq=5000 \
  --log_freq=500 --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=4
SRC=$OUT/checkpoints/010000/pretrained_model
$PY -c "import json;c=json.load(open(\"$SRC/train_config.json\"));c.setdefault(\"wandb\",{})[\"enable\"]=False;c[\"wandb\"].pop(\"add_tags\",None);json.dump(c,open(\"/tmp/cd_clair.json\",\"w\"))"
COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=/tmp/cd_clair.json --resume=false --policy.pretrained_path=$SRC \
  --output_dir=$OUT/cooldown --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0
echo "[$(date +%H:%M:%S)] ===== PLAFOND CLAIR TERMINÉ ====="
