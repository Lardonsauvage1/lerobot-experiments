#!/bin/bash
# TEST DÉCISIF "l'archi est-elle maxée ?" — cooldown (annealing) du joint depuis le plateau constant.
# WSD/WSM : un vrai decay LR 1e-4->0 depuis le plateau devrait settler le minimum.
# Question : dépasse-t-il le SWA/merge (late10 = 88 %, même région 32-50k) ?
#   > 88 % -> l'archi avait de la marge laissée sur la table (le post-hoc masquait).
#   ~ 88 % -> le post-hoc (SWA/ensemble) avait déjà extrait le max -> prochain plafond = init (B) / param (A).
# Charge les poids de 50k (pretrained_path, optimiseur FRAIS, step=0), pas de resume (ne touche PAS joint_r34).
# Tout sur Principal (train + éval, pas de transfert). En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/79_cooldown_test.sh > /tmp/cooldown.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RUN=results/runs/can/joint_r34_bigunet
SRC=$RUN/checkpoints/050000/pretrained_model
OUT=results/runs/can/joint_cooldown_50k
N_CD=5000

echo "[79] $(date '+%H:%M') attente Principal libre (éval en cours ?)..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|47_eval_joint_ensemble|50_train_valloss" >/dev/null; do sleep 60; done

# Le train_config.json du 50k a un champ wandb.add_tags que draccus refuse -> on le sanitize.
CFG=/tmp/cooldown_config.json
$PY -c "import json; c=json.load(open('$SRC/train_config.json')); c.get('wandb',{}).pop('add_tags',None); c.setdefault('wandb',{})['enable']=False; json.dump(c,open('$CFG','w'))"
echo "[79] config sanitizé -> $CFG"

echo "[79] $(date '+%H:%M') COOLDOWN depuis 50k : LR 1e-4 -> 0 sur ${N_CD} steps"
COOLDOWN_STEPS=$N_CD COOLDOWN_LR0=1e-4 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$CFG --resume=false \
  --policy.pretrained_path=$SRC \
  --output_dir=$OUT --steps=$N_CD --save_freq=1000 --log_freq=50 --eval_freq=0

echo "[79] $(date '+%H:%M') éval des checkpoints cooldown (n=50, kp=50)"
for s in 1000 2000 3000 4000 5000; do
  [ -s "$OUT/checkpoints/$(printf '%06d' $s)/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$OUT" --steps "$s" --n 50 --kp 50 --out "$OUT/rollouts_50.csv"
done

echo "[79] DONE — cooldown vs SWA late10 (88 %) :"
cat "$OUT/rollouts_50.csv" 2>/dev/null
