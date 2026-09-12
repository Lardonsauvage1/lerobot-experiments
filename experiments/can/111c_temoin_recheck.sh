#!/bin/bash
# CONTRÔLE MANQUANT : rejouer le témoin (radius=0) AVEC enregistrement des trajectoires.
#
# Pourquoi c'est indispensable. Les échecs à 4 cm font 3,70 cycles descente/remontée. Mais
# le témoin a tourné avant que l'enregistrement existe, donc on ignore combien de cycles
# font SES échecs (31 % des épisodes). Sans ce chiffre on ne peut PAS attribuer la boucle à
# l'occlusion : si le modèle bouclait déjà sans occlusion, la boucle serait un trait du
# modèle et non un effet de la perte de vue. C'est le contrôle qui rend la conclusion valide.
#
# Bonus gratuit : le succès doit retomber sur ~68,8 % — seconde vérification de reproductibilité.
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
L=results/logs/can/111_occlusion_curve.log
while pgrep -f "111_occlusion_curve.py" > /dev/null; do sleep 30; done
echo "[temoin] courbe terminée -> rejeu du témoin AVEC enregistrement (contrôle des cycles)" >> "$L"
OCC_RADII=0 OCC_OUT=results/runs/can/occluded/temoin_recheck.json \
  caffeinate -i venv312/bin/python -u experiments/can/111_occlusion_curve.py >> "$L" 2>&1
echo "[temoin] contrôle terminé -> traj_r00.npz disponible" >> "$L"
