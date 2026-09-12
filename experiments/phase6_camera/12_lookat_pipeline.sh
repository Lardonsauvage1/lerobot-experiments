#!/bin/bash
# Pipeline look-at : génère le dataset look-at, vérifie son intégrité, puis enchaîne l'entraînement.
# Lancer (le caffeinate empêche la veille pendant gen + ~12h de train) :
#   nohup caffeinate -i bash experiments/phase6_camera/12_lookat_pipeline.sh > /tmp/lookat_pipeline.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
DS=data_cache/lerobot_can_ph_proprio_birdview_lookat
GENLOG=results/logs/phase6_camera/gen_lookat.log
mkdir -p results/logs/phase6_camera

echo "[12] $(date '+%H:%M %d/%m') — génération dataset look-at..."
$PY -u experiments/phase6_camera/10_make_camaug_lookat.py > "$GENLOG" 2>&1
if [ ! -s "$DS/meta/info.json" ]; then
  echo "[12] ⚠️ génération ÉCHOUÉE (pas de $DS/meta/info.json) — entraînement NON lancé. Fin du log gen :"
  tail -8 "$GENLOG"; exit 1
fi
echo "[12] ✓ dataset prêt : $(grep -o '\"total_frames\":[0-9]*' $DS/meta/info.json 2>/dev/null) — lancement entraînement"

bash experiments/phase6_camera/11_train_lookat_r34_bigunet.sh
echo "[12] PIPELINE LOOK-AT TERMINÉ"
