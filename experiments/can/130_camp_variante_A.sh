#!/usr/bin/env bash
# VARIANTE A — fidèle au papier CAMP : pré-entraînement -> warm-up gelé -> FINETUNING CONJOINT.
#
# La variante B (module gelé, codes précalculés) a donné un résultat NÉGATIF propre :
# 60,0 % vs 56,0 % pour la baseline, McNemar p = 0,539 (36 gains / 30 pertes = bruit pur),
# et les descentes à vide ne baissent pas. Mais B n'est pas CAMP : le papier finetune le
# LSTM avec la policy (lr x alpha = 0,1). C'est cette phase qu'on ajoute ici.
#
# Le warm-up existe déjà : `b3_camp` = 10k steps avec mémoire GELÉE. On repart de là.
# Coût du déroulement complet mesuré : +66 ms/step sur ~950 ms, soit +7 % — j'avais renoncé
# à cette variante en surestimant ce coût.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/130_camp_A.log
MEM0=results/runs/can/camp/memory_L64_K32_m32.pt
WARM=results/runs/can/b3_camp/checkpoints/010000/pretrained_model
OUT=results/runs/can/b3_campA
step() { echo "" >> $LOG; echo "[$(date '+%H:%M:%S')] ═══ $* ═══" >> $LOG; }

[ -s "$WARM/model.safetensors" ] || { echo "warm-up absent" >> $LOG; exit 1; }

step "FINETUNING CONJOINT 5k steps (LSTM dégelé, lr x 0,1) depuis le warm-up"
rm -rf "$OUT"
$PY -c "import json;c=json.load(open('$WARM/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('/tmp/ft_A.json','w'))"
CAMP=1 CAMP_FINETUNE=1 CAMP_ALPHA=0.1 CAMP_CKPT=$MEM0 \
$PY -u experiments/lift/50_train_valloss.py --config_path=/tmp/ft_A.json --resume=false \
  --policy.pretrained_path="$WARM" --output_dir="$OUT" \
  --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0 >> $LOG 2>&1

MEM1=$OUT/checkpoints/005000/memory_finetuned.pt
[ -s "$MEM1" ] || { step "❌ LSTM finetuné absent -> arrêt"; exit 2; }
$PY -c "
import torch
a=torch.load('$MEM0',weights_only=False)['state_dict']; b=torch.load('$MEM1',weights_only=False)['state_dict']
d=max(float((a[k].cpu()-b[k].cpu()).abs().max()) for k in a)
print(f'[camp-A] ecart max des poids LSTM apres finetuning : {d:.3e}')" >> $LOG 2>&1

step "COOLDOWN (LSTM finetuné conservé, toujours déroulé)"
SRC=$OUT/checkpoints/005000/pretrained_model
$PY -c "import json;c=json.load(open('$SRC/train_config.json'));c.setdefault('wandb',{})['enable']=False;c['wandb'].pop('add_tags',None);json.dump(c,open('/tmp/cd_A.json','w'))"
CAMP=1 CAMP_FINETUNE=1 CAMP_ALPHA=0.1 CAMP_CKPT=$MEM1 \
COOLDOWN_STEPS=5000 COOLDOWN_LR0=1e-4 \
$PY -u experiments/lift/50_train_valloss.py --config_path=/tmp/cd_A.json --resume=false \
  --policy.pretrained_path="$SRC" --output_dir=$OUT/cooldown \
  --steps=5000 --save_freq=5000 --log_freq=500 --eval_freq=0 >> $LOG 2>&1

MEM2=$OUT/cooldown/checkpoints/005000/memory_finetuned.pt
[ -s "$MEM2" ] || MEM2=$MEM1
step "ÉVAL 150 rollouts @3cm — mémoire FINETUNÉE déroulée en ligne"
$PY -u experiments/can/120_eval_occluded.py \
  --ckpt $OUT/cooldown/checkpoints/005000/pretrained_model \
  --radius 0.03 --n 150 --camp "$MEM2" --tag b3_campA >> $LOG 2>&1
step "VARIANTE A TERMINÉE"
