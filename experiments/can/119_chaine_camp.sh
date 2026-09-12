#!/usr/bin/env bash
# CHAÎNE DÉCISIVE : la mémoire CAMP améliore-t-elle une policy sous occlusion ?
#
#   bench -> baseline 9D -> cooldown -> éval  ||  CAMP 41D -> cooldown -> éval
#
# Le SEUL écart entre les deux bras est `CAMP=1` (32 dimensions de plus dans le
# conditionnement). Même dataset occlus à 3 cm, même archi, même budget, même seed.
# C'est cette égalité qui rend l'écart interprétable.
#
# ⚠️ Budget volontairement réduit (20k × batch 16 = 320k échantillons, contre 960k pour le
# témoin wristcap_A) : les DEUX bras seront sous-entraînés par rapport au témoin, mais à
# budget identique — on mesure l'effet de la mémoire, pas le plafond absolu.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/119_chaine.log
MEM=results/runs/can/camp/memory_L64_K32_m32.pt
DS=data_cache/lerobot_can_ph_proprio_occ03
RID=local/can_ph_proprio_occ03
STEPS=${STEPS:-20000}
EPS=$($PY -c "print(list(range(150)))" | tr -d ' ')
step() { echo "" >> $LOG; echo "[$(date '+%m-%d %H:%M:%S')] ═══ $* ═══" >> $LOG; }

train() { # $1=nom  $2=camp(0/1)
  local OUT=results/runs/can/$1
  rm -rf "$OUT"
  export EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4
  if [ "$2" = "1" ]; then export CAMP=1 CAMP_CKPT=$MEM; else unset CAMP CAMP_CKPT; fi
  $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.vision_backbone=resnet34 "--policy.down_dims=[128,256,512]" \
    --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=$RID --dataset.root=$DS --dataset.episodes="$EPS" \
    --output_dir="$OUT" --batch_size=16 --steps=$STEPS --save_freq=5000 --log_freq=500 \
    --eval_freq=0 --seed=42 --wandb.enable=false --num_workers=2 >> $LOG 2>&1
}

cooldown() { # $1=nom  $2=camp(0/1)   — LR 1e-4 -> 0 sur 5k (règle du projet : jamais évaluer le brut)
  local SRC=results/runs/can/$1/checkpoints/$(printf '%06d' $STEPS)/pretrained_model
  [ -s "$SRC/model.safetensors" ] || SRC=results/runs/can/$1/checkpoints/last/pretrained_model
  [ -s "$SRC/model.safetensors" ] || { echo "!! pas de ckpt pour $1" >> $LOG; return 1; }
  local CFG=/tmp/cd_$1.json
  $PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('$CFG','w'))"
  export EMA=1 COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4
  if [ "$2" = "1" ]; then export CAMP=1 CAMP_CKPT=$MEM; else unset CAMP CAMP_CKPT; fi
  $PY -u experiments/lift/50_train_valloss.py --config_path=$CFG --resume=false \
    --policy.pretrained_path="$SRC" --output_dir=results/runs/can/$1/cooldown \
    --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0 >> $LOG 2>&1
  unset COOLDOWN_STEPS COOLDOWN_LR0
}

evaluate() { # $1=nom  $2=camp(0/1)
  unset CAMP CAMP_CKPT EMA CONST_LR CONST_LR_VALUE
  local CK=results/runs/can/$1/cooldown/checkpoints/005000/pretrained_model
  [ -s "$CK/model.safetensors" ] || { echo "!! pas de cooldown pour $1" >> $LOG; return 1; }
  local EXTRA=""; [ "$2" = "1" ] && EXTRA="--camp $MEM"
  # 150 et non 250 : l'éval du 09/09 a pris 5 h 30 parce qu'un ÉCHEC consomme les 300 pas
  # alors qu'une réussite s'arrête tôt — plus un modèle rate, plus il coûte cher à mesurer.
  # IC95 ±8 pts à n=150, largement assez pour trancher un écart A/B.
  $PY -u experiments/can/120_eval_occluded.py --ckpt "$CK" --radius 0.03 --n 150 \
      --tag "$1" $EXTRA >> $LOG 2>&1
}

# Relance du 2026-09-09 23:5x après correction du banc (occlusion qui ne se coupait jamais
# après la saisie). Le bras A précédent (2,8 %) mesurait ce défaut, pas la baseline -> jeté.
step "BENCH sauté (vitesse déjà mesurée : 1,52-1,63 step/s à batch 16)"

step "BRAS A — baseline 9D (sans mémoire) : entraînement"
train occ03_baseline 0
step "BRAS A — cooldown"; cooldown occ03_baseline 0
step "BRAS A — éval 250 rollouts @ 3 cm"; evaluate occ03_baseline 0

step "BRAS B — CAMP-lite 41D : entraînement"
train occ03_camp 1
step "BRAS B — cooldown"; cooldown occ03_camp 1
step "BRAS B — éval 250 rollouts @ 3 cm (mémoire EN LIGNE)"; evaluate occ03_camp 1

step "RÉSULTAT"
$PY - >> $LOG 2>&1 <<'PYEOF'
import json, pathlib
p = pathlib.Path("results/runs/can/occluded")
for t in ["occ03_baseline", "occ03_camp"]:
    f = p / f"eval_{t}.json"
    if f.exists():
        d = json.load(open(f))
        print(f"{t:>16} : {d['n_success']}/{d['n']} = {d['success_rate']:.1%} "
              f"[{d['ci95'][0]:.1%}-{d['ci95'][1]:.1%}] | cycles/échec {d['z_cycles_failed']}")
    else:
        print(f"{t:>16} : MANQUANT")
print("\nTémoin markovien entraîné sur CLAIR, évalué @3cm : 37,6 % (rappel)")
PYEOF
step "CHAÎNE TERMINÉE"
