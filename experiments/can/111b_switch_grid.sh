#!/bin/bash
# Bascule vers la grille révisée SANS perdre le rayon 4 cm en cours (250 rollouts presque finis).
# Le script 111 ne sauve son JSON qu'à la FIN d'un rayon -> on attend que 0.04 y soit écrit,
# puis on coupe (sinon il enchaînerait sur 0.07, l'ancienne liste étant figée en mémoire)
# et on relance : le resume saute 0.0 et 0.04 déjà faits et attaque 1, 2, 3 cm.
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
J=results/runs/can/occluded/occlusion_curve.json
L=results/logs/can/111_occlusion_curve.log
OLD=$1
while ! grep -q '"radius": 0.04' "$J" 2>/dev/null; do
  kill -0 "$OLD" 2>/dev/null || { echo "[switch] le run s'est arrêté seul avant 0.04" >> "$L"; break; }
  sleep 20
done
echo "[switch] rayon 4 cm sauvegardé -> bascule sur la grille 1/2/3 cm + inf" >> "$L"
kill "$OLD" 2>/dev/null; sleep 8; kill -9 "$OLD" 2>/dev/null; sleep 3
nohup caffeinate -i venv312/bin/python -u experiments/can/111_occlusion_curve.py >> "$L" 2>&1 &
echo "[switch] relancé PID=$!" >> "$L"
