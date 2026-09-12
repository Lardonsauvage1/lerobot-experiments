#!/bin/bash
# SWA : pour chaque combinaison de checkpoints joint -> moyenne des poids -> rollout n=50 (kp=50).
# Construit une table : combo | steps membres | taux individuels | taux de la moyenne des poids.
# Compositions : best-only, best-large, null-only, best+null, mediums-only, late-window classique, random.
# IDEMPOTENT (skip les combos déjà dans la table) + cède le Principal si une autre éval tourne.
# Lancer : nohup bash experiments/can/65_swa_eval.sh > /tmp/swa_eval.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RDJ=results/runs/can/joint_r34_bigunet
TABLE=$RDJ/swa_table.csv
[ -s "$TABLE" ] || echo "combo,membres,taux_individuels,n,taux_moyenne_poids,ci95_lo,ci95_hi" > "$TABLE"

run_combo () {
  local name="$1" steps="$2"
  grep -q "^$name," "$TABLE" && { echo "[65] $name déjà fait, skip"; return; }
  # cède si une AUTRE éval occupe le Principal (run31-jumeau=32_eval_dense, camshift=13_eval)
  while pgrep -f "32_eval_dense_birdview|13_eval_camshift" >/dev/null; do echo "[65] Principal occupé, attente..."; sleep 60; done
  echo "[65] $(date '+%H:%M') combo $name : $steps"
  local out=/tmp/swa_$name
  $PY experiments/can/65_swa_make.py --steps "$steps" --out "$out" || { echo "[65] make échoué $name"; return; }
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$out" --steps 0 --n 50 --kp 50 --out "$out/r.csv" || { echo "[65] eval échoué $name"; return; }
  $PY -c "
import csv
r=list(csv.DictReader(open('$out/r.csv')))[-1]
avg=float(r['success_rate'])*100; lo=float(r.get('ci95_low',0) or 0)*100; hi=float(r.get('ci95_high',0) or 0)*100
by={int(float(x['step'])):float(x['success_rate'])*100 for x in csv.DictReader(open('$RDJ/rollouts_50.csv'))}
steps=[int(s) for s in '$steps'.split(',')]
rates='|'.join(f'{by.get(s,-1):.0f}' for s in steps)
mem='|'.join(f'{s//1000}k' for s in steps)
open('$TABLE','a').write(f'$name,{mem},{rates},{len(steps)},{avg:.0f},{lo:.0f},{hi:.0f}\n')
print(f'[65] $name -> MOYENNE DES POIDS = {avg:.0f}%  (membres: {rates})')
"
  rm -rf "$out"
}

# === compositions ===
run_combo best3      "30000,24000,46000"            # les 3 meilleurs (56,48,48)
run_combo best5      "30000,24000,46000,50000,18000" # les 5 meilleurs
run_combo null3      "34000,42000,20000"            # 3 checkpoints nuls (0,0,0)
run_combo best_null  "30000,34000"                  # meilleur + nul (56,0) : rattrape ou détruit ?
run_combo med3       "32000,48000,36000"            # 3 moyens (34,34,30)
run_combo late5      "42000,44000,46000,48000,50000" # SWA classique : 5 derniers (0,38,48,34,46)
run_combo rand5      "22000,34000,44000,28000,16000" # aléatoire (12,0,38,18,0)
# --- bonne taille SWA (5-10), pour ne pas conclure que sur des combos trop petites ---
run_combo late10     "32000,34000,36000,38000,40000,42000,44000,46000,48000,50000" # 10 derniers (queue LR constant)
run_combo goodonly9  "18000,24000,30000,32000,36000,44000,46000,48000,50000"        # les 9 >=30% (sans les nuls)

echo "[65] DONE — table SWA :"; column -t -s, "$TABLE"
