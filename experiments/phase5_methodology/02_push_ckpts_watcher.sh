#!/bin/bash
# Phase 5 — Watcher Mac qui pousse vers atomman les nouveaux ckpts via tar|ssh (openrsync macOS buggy).
# Détecte les nouveaux dossiers checkpoints/XXXXXX/ et les pousse immédiatement.
# Lancer : nohup bash experiments/phase5_methodology/02_push_ckpts_watcher.sh > results/logs/phase5_methodology/run_02_push_watcher.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
LOCAL_CKPTS="results/runs/phase5_methodology/26_resnet34_dense/checkpoints"
REMOTE_ROOT='/home/sam/Documents/projet VScode/experience_Le/results/runs/phase5_methodology/26_resnet34_dense'
PUSHED_LIST="/tmp/02_pushed_steps.txt"
> "$PUSHED_LIST"

ssh atomman "mkdir -p \"$REMOTE_ROOT/checkpoints\""
echo "[02] $(date '+%H:%M:%S') watcher démarré (tar|ssh)"

# Attendre que le dir checkpoints existe (le train le crée)
while [ ! -d "$LOCAL_CKPTS" ]; do sleep 30; done

while true; do
  for ckpt in "$LOCAL_CKPTS"/*/; do
    step=$(basename "$ckpt")
    # ignore non-numérique (ex: 'last' symlink)
    case "$step" in *[!0-9]*) continue;; esac
    # PUSH SEULEMENT step % 1000 == 0 (ckpts utilisés par worker 03 sur atomman).
    # Strip leading zeros for arithmetic comparison.
    step_int=$((10#$step))
    [ $((step_int % 1000)) -ne 0 ] && continue
    # déjà poussé ?
    grep -qx "$step" "$PUSHED_LIST" 2>/dev/null && continue
    # vérifie complétude
    [ ! -d "$ckpt/training_state" ] && continue
    [ ! -f "$ckpt/pretrained_model/model.safetensors" ] && continue

    echo "[02] $(date '+%H:%M:%S') push step=$step (pretrained_model only)"
    # tar | gzip | ssh : seulement pretrained_model (~190 MB) pas training_state (~380 MB)
    # gzip ajoute ~5 % CPU mais réduit ~10 % de transfert. À 3 Mbps c'est rentable.
    if (cd "$LOCAL_CKPTS/$step" && tar czf - pretrained_model) | \
         ssh atomman "mkdir -p \"$REMOTE_ROOT/checkpoints/$step\" && cd \"$REMOTE_ROOT/checkpoints/$step\" && tar xzf -"; then
      echo "$step" >> "$PUSHED_LIST"
    else
      echo "[02] $(date '+%H:%M:%S') ⚠ push échec step=$step"
    fi
  done

  # train fini ? -> dernière passe puis exit
  if ! pgrep -f "50_train_valloss" >/dev/null 2>&1; then
    sleep 60
    LOCAL_N=$(ls -d "$LOCAL_CKPTS"/*/ 2>/dev/null | wc -l | tr -d ' ')
    PUSHED_N=$(wc -l < "$PUSHED_LIST" | tr -d ' ')
    # NB: LOCAL_N inclut 'last' (symlink) -> on accepte si PUSHED_N >= LOCAL_N-1
    if [ "$PUSHED_N" -ge "$((LOCAL_N - 1))" ]; then
      echo "[02] $(date '+%H:%M:%S') train fini, $PUSHED_N poussés (sur $LOCAL_N) — fin"
      exit 0
    fi
  fi
  sleep 30
done
