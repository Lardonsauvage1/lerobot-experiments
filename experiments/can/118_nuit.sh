#!/usr/bin/env bash
# Enchaînement automatique de la nuit du 2026-09-09 (Sam dort ; gb10 et mac2 injoignables,
# donc TOUT sur le Mac principal, en séquentiel — un seul rendu mujoco à la fois).
#
# NE LANCE AUCUN ENTRAÎNEMENT LONG : un run Can complet = 20-40 h sur M1 et mobiliserait la
# machine plusieurs jours, décision qui revient à Sam une fois qu'il aura vu l'étape 0. Ce
# script fait tout le reste, et surtout VALIDE LE PIPELINE CAMP de bout en bout (smoke court)
# pour qu'aucun bug de plomberie ne se découvre après 30 h de calcul.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/118_nuit.log
step() { echo "" >> $LOG; echo "[$(date '+%H:%M:%S')] ═══ $* ═══" >> $LOG; }
run()  { echo "[$(date '+%H:%M:%S')] \$ $*" >> $LOG; "$@" >> $LOG 2>&1; echo "[$(date '+%H:%M:%S')] exit=$?" >> $LOG; }

step "ATTENTE : courbe d'occlusion + contrôle témoin"
# 111c_temoin_recheck.sh relance 111 pour le témoin -> on attend le fichier qu'il produit,
# pas seulement l'absence de process (sinon on démarrerait dans la fenêtre entre les deux).
DEADLINE=$(( $(date +%s) + 4*3600 ))
while [ ! -f results/runs/can/occluded/traj_r00.npz ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
while pgrep -f "111_occlusion_curve.py" >/dev/null && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
[ -f results/runs/can/occluded/traj_r00.npz ] || echo "!! traj_r00.npz absent (contrôle témoin KO) — on continue sans l'analyse appariée" >> $LOG

step "ANALYSES de l'étape 0"
run $PY -u experiments/can/115_diag_failure_mode.py
run $PY -u experiments/can/116_plot_occlusion_curve.py
run $PY -u experiments/can/117_paired_analysis.py

step "CAMP phase 1 : pré-entraînement du module mémoire (4000 steps)"
run $PY -u experiments/can/113_camp_pretrain.py --steps 4000

step "CAMP : LE test — la mémoire distingue-t-elle les tentatives ?"
run $PY -u experiments/can/114_camp_discriminates.py --ckpt results/runs/can/camp/memory_L64_K32_m32.pt

step "CHOIX DU RAYON pour l'étape 0b"
R=$($PY - <<'PYEOF'
import json
res = json.load(open("results/runs/can/occluded/occlusion_curve.json"))["results"]
base = next(r["success_rate"] for r in res if r["radius"] == 0)
# On veut le rayon qui fait chuter FRANCHEMENT sans annuler la tâche : trop peu de chute et
# on ne peut rien mesurer ; plancher à 0 et il ne reste plus rien à récupérer par la mémoire.
# Cible = la moitié du témoin, en n'acceptant que la plage [20 %, 70 %] du témoin.
cands = [r for r in res if 0 < r["radius"] < 1e3 and 0.20 * base <= r["success_rate"] <= 0.70 * base]
pool = cands or [r for r in res if 0 < r["radius"] < 1e3]
print(min(pool, key=lambda r: abs(r["success_rate"] - 0.5 * base))["radius"])
PYEOF
)
echo "[$(date '+%H:%M:%S')] rayon retenu = $R m" >> $LOG
echo "$R" > results/runs/can/occluded/chosen_radius.txt

step "ÉTAPE 0b : datasets Can OCCLUS (rayon $R)"
# 9D = baseline vision pure (mêmes entrées que le témoin) ; 12D = ORACLE (can_pos donnée
# même occluse) -> l'écart entre les deux borne ce que la mémoire peut au mieux rapporter.
run $PY -u src/can_to_lerobot.py --proprio --occlude "$R"
run $PY -u src/can_to_lerobot.py --occlude "$R"

step "VALIDATION du pipeline CAMP de bout en bout (300 steps, jetable)"
# But : prouver que CAMP=1 charge la mémoire gelée, étend state 9D->41D, étend les stats et
# que la policy s'entraîne. 300 steps suffisent — on ne cherche PAS un modèle, on cherche
# l'absence de bug avant d'engager 30 h.
OCCTAG=$($PY -c "print(f'_occ{int(round(float(\"$R\")*100)):02d}')")
DS=data_cache/lerobot_can_ph_proprio${OCCTAG}
if [ -d "$DS" ]; then
  EPS=$($PY -c "print(list(range(20)))" | tr -d ' ')
  rm -rf results/runs/can/camp_smoke
  # export explicite : préfixer un appel de FONCTION bash par des affectations ne propage
  # pas l'environnement de façon fiable (contrairement à une commande externe).
  export CAMP=1 CAMP_CKPT=results/runs/can/camp/memory_L64_K32_m32.pt
  export EMA=1 CONST_LR=1 CONST_LR_VALUE=1e-4
  run $PY -u experiments/lift/50_train_valloss.py \
    --policy.type=diffusion --policy.vision_backbone=resnet34 --policy.down_dims=[128,256,512] \
    --policy.device=mps --policy.push_to_hub=false --policy.crop_shape=null \
    --dataset.repo_id=local/can_ph_proprio${OCCTAG} --dataset.root=$DS --dataset.episodes=$EPS \
    --output_dir=results/runs/can/camp_smoke \
    --batch_size=8 --steps=300 --save_freq=300 --log_freq=50 --eval_freq=0 \
    --seed=42 --wandb.enable=false --num_workers=2
  unset CAMP CAMP_CKPT EMA CONST_LR CONST_LR_VALUE
else
  echo "!! dataset $DS absent -> validation CAMP sautée" >> $LOG
fi

step "TERMINÉ"
