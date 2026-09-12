#!/usr/bin/env bash
# Enchaînement APRÈS la chaîne A/B : recolle la référence et produit le verdict.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
LOG=results/logs/can/123_post.log
TEMOIN=results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model
step() { echo "" >> $LOG; echo "[$(date '+%m-%d %H:%M:%S')] ═══ $* ═══" >> $LOG; }

step "ATTENTE de la fin de la chaîne A/B"
while pgrep -f "119_chaine_camp" >/dev/null; do sleep 120; done

# Le témoin n'a jamais été mesuré sur le banc CORRIGÉ : la courbe d'hier (37,6 % @3cm) a été
# obtenue avec l'occlusion qui ne se coupait jamais après la saisie. Sans ce point, A et B
# sont comparables entre eux mais raccrochables à rien.
step "RÉFÉRENCE : témoin (entraîné sur CLAIR) sur le banc corrigé, 150 rollouts @3cm"
$PY -u experiments/can/120_eval_occluded.py --ckpt "$TEMOIN" --radius 0.03 --n 150 \
    --tag temoin_bancfix >> $LOG 2>&1

step "GRAPHE + VERDICT"
$PY -u experiments/can/125_plot_ab.py >> $LOG 2>&1

step "RÉCAPITULATIF"
$PY - >> $LOG 2>&1 <<'PYEOF'
import json, pathlib
p = pathlib.Path("results/runs/can/occluded")
rows = []
for t, lab in [("temoin_bancfix", "témoin (entraîné CLAIR)"),
               ("occ03_baseline", "baseline 9D (OCCLUS)"),
               ("occ03_camp", "CAMP-lite 41D")]:
    f = p / f"eval_{t}.json"
    rows.append((lab, json.load(open(f))) if f.exists() else (lab, None))
for lab, d in rows:
    if d:
        print(f"{lab:>26} : {d['n_success']}/{d['n']} = {d['success_rate']:6.1%} "
              f"[{d['ci95'][0]:.1%}-{d['ci95'][1]:.1%}] | cycles/échec {d.get('z_cycles_failed')}")
    else:
        print(f"{lab:>26} : MANQUANT")
b = next((d for l, d in rows if d and "baseline" in l), None)
c = next((d for l, d in rows if d and "CAMP" in l), None)
if b and c:
    delta = (c["success_rate"] - b["success_rate"]) * 100
    # écarts d'IC : verdict prudent, un chevauchement ne tranche pas
    sep = c["ci95"][0] > b["ci95"][1] or b["ci95"][0] > c["ci95"][1]
    print(f"\nÉCART CAMP - baseline : {delta:+.1f} points  "
          f"({'IC95 DISJOINTS -> écart net' if sep else 'IC95 qui se recouvrent -> non concluant à ce n'})")
PYEOF
step "POST-CHAÎNE TERMINÉE"
