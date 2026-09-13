#!/bin/bash
# Recuit de CHAQUE checkpoint de la phase a LR constant, du plus avance au moins avance.
#
# But : comparer des modeles COMPARABLES. Jusqu'ici on opposait un modele recuit a des
# checkpoints bruts, ce qui melange deux effets -- la duree d'entrainement et le recuit.
# En recuisant chacun, on isole la duree.
#
# Attentes par MARQUEUR (existence d'un fichier), jamais par pgrep -f : ce motif matche
# le shell qui a cree le script, piege qui a coute 7 heures le 2026-09-09.
set -u
BASE=$HOME/lerobot-experiments/outputs
SRC=$BASE/pince_const_20260909/checkpoints
JOURNAL=$HOME/chaine_cooldowns.log

echo "$(date +%H:%M) attente de la fin du cooldown en cours (036000)" >> $JOURNAL
until [ -d "$BASE/pince_cooldown_20260909/checkpoints/004000/pretrained_model" ]; do
  sleep 60
done
echo "$(date +%H:%M) cooldown 036000 termine" >> $JOURNAL

for PAS in 030000 024000 018000 012000 006000; do
  OUT=$BASE/pince_cooldown_$PAS
  if [ -d "$OUT/checkpoints/004000/pretrained_model" ]; then
    echo "$(date +%H:%M) $PAS deja fait, saute" >> $JOURNAL; continue
  fi
  CFG=$HOME/train_config_cooldown_$PAS.json
  python3 - "$PAS" "$CFG" "$OUT" <<'PY'
import json, sys
pas, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open("/home/sam/train_config_pince_cooldown.json"))
c["policy"]["pretrained_path"] = (
    "/home/sam/lerobot-experiments/outputs/pince_const_20260909/checkpoints/%s/pretrained_model" % pas)
c["output_dir"] = out
c["steps"] = 4000
c["save_freq"] = 4000          # seul le point FINAL nous interesse
json.dump(c, open(cfg, "w"), indent=1)
PY
  echo "$(date +%H:%M) demarrage du recuit de $PAS" >> $JOURNAL
  ( set +u   # les scripts d'environnement sources (ROS setup.bash, oneAPI) lisent des
             # variables non definies ; sous `set -u` le sous-shell meurt avant le premier
             # pas d'entrainement. Mesure du 2026-09-10 : 5 recuits echoues en 0 s.
    cd $HOME/lerobot-experiments \
    && source $HOME/roby_xpu_env.sh \
    && source /opt/ros/jazzy/setup.bash \
    && $HOME/ipex_test_venv/bin/python -m lerobot.scripts.lerobot_train \
         --config_path=$CFG ) > $HOME/cooldown_$PAS.log 2>&1
  if [ -d "$OUT/checkpoints/004000/pretrained_model" ]; then
    echo "$(date +%H:%M) $PAS OK" >> $JOURNAL
  else
    echo "$(date +%H:%M) $PAS ECHEC -- voir ~/cooldown_$PAS.log" >> $JOURNAL
  fi
done
echo "$(date +%H:%M) CHAINE TERMINEE" >> $JOURNAL
