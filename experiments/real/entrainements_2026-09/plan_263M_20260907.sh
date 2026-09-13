#!/usr/bin/env bash
# Enchainement automatique du 263M sur le dataset propre (demande Sam, 2026-09-07).
#
#   phase 1  06:30 -> 09:30   entrainement 3 h        (deja lance a part)
#   cooldown 09:30 -> 10:00   depuis le checkpoint de phase 1 -> modele utilisable
#   phase 2  10:00 -> 18:30   REPREND le checkpoint AVANT cooldown, LR constant
#   cooldown 18:30 -> 19:00   depuis le checkpoint de phase 2 -> modele final
#
# --init-from <dir> relit <dir>/263M/pretrained_model : la phase 2 repart donc bien du
# checkpoint d'entrainement, PAS du cooldown. ⚠️ Seuls les POIDS sont repris, l'etat de
# l'optimiseur redemarre a zero (c'est ainsi que le balayage de juillet faisait deja).
set -u
BASE="$HOME/lerobot-experiments/outputs"
DS="$HOME/lerobot-experiments/data_cache/lerobot_apple_propre_2cam_128"
P1="$BASE/propre_263M_20260907"
CD1="$BASE/propre_263M_cooldown1"
P2="$BASE/propre_263M_phase2"
CD2="$BASE/propre_263M_cooldown2"
LOG="$HOME/plan_263M.log"
COMMUN="--root $DS --only 263M --batch 64 --save-every 15 --metrics-every 5 --snap-every 15"

dit() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

dit "=== plan demarre ; attente de la fin de la phase 1 (PID $1) ==="
while kill -0 "$1" 2>/dev/null; do sleep 30; done
dit "phase 1 terminee."

dit "--- cooldown 1 (30 min) depuis le checkpoint de phase 1 ---"
rm -rf "$CD1"
bash "$HOME/roby_train_xpu.sh" $COMMUN --out "$CD1" --hours 0.5 \
     --lr-schedule cosine --init-from "$P1" >> "$LOG" 2>&1
dit "cooldown 1 termine -> $CD1"

# Duree de la phase 2 : jusqu'a 18h30 aujourd'hui, calculee au dernier moment.
FIN=$(date -d "today 18:30" +%s); MAINTENANT=$(date +%s)
H=$(awk "BEGIN{printf \"%.3f\", ($FIN-$MAINTENANT)/3600}")
dit "--- phase 2 : $H h d'entrainement, reprise du checkpoint AVANT cooldown ---"
if awk "BEGIN{exit !($H > 0.1)}"; then
  rm -rf "$P2"
  bash "$HOME/roby_train_xpu.sh" $COMMUN --out "$P2" --hours "$H" \
       --lr-schedule constant --init-from "$P1" >> "$LOG" 2>&1
  dit "phase 2 terminee -> $P2"

  dit "--- cooldown 2 (30 min) depuis le checkpoint de phase 2 ---"
  rm -rf "$CD2"
  bash "$HOME/roby_train_xpu.sh" $COMMUN --out "$CD2" --hours 0.5 \
       --lr-schedule cosine --init-from "$P2" >> "$LOG" 2>&1
  dit "cooldown 2 termine -> $CD2"
else
  dit "!! il est deja trop tard pour la phase 2 ($H h) -> ignoree"
fi
dit "=== PLAN TERMINE ==="
