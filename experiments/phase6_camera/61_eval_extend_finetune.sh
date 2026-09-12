#!/bin/bash
# ÉVAL des rollouts réguliers (n=100) des nouveaux checkpoints du fine-tune prolongé (16k->30k tous les 2k).
# Attend (a) que la prolongation mac2 soit FINIE (30000/30000) et (b) que le Principal soit LIBRE
# (run31-jumeau eval terminée). Transfère les checkpoints depuis mac2, évalue, régénère la frise.
# En FICHIER. Lancer : nohup bash experiments/phase6_camera/61_eval_extend_finetune.sh > /tmp/ext_ft_eval.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP=100.110.237.53
RUN=results/runs/phase6_camera/lookat_finetune_run31

echo "[61] $(date '+%H:%M') attente fin de la prolongation mac2 (30000/30000)..."
while true; do
  done_mac2=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 \
    "pgrep -f '50_train_valloss.*lookat_finetune_run31' >/dev/null && echo RUN || (grep -aoE '30000/30000' /tmp/run_lookat_ft_ext.log 2>/dev/null | tail -1)" 2>/dev/null)
  [ "$done_mac2" = "30000/30000" ] && break
  sleep 180
done
echo "[61] $(date '+%H:%M') prolongation finie. Attente Principal libre (run31-jumeau eval finie)..."
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift" >/dev/null; do sleep 120; done
echo "[61] $(date '+%H:%M') Principal libre -> transfert + éval n=100 des nouveaux checkpoints."

for s in 016000 018000 020000 022000 024000 026000 028000 030000; do
  CK=$RUN/checkpoints/$s/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RUN/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] && \
    $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RUN" --steps "${s#0}" --n 100 --out "$RUN/rollouts_100.csv"
done

echo "[61] éval finie -> régénération frise run31 -> fine-tune (étendue) + 4-panneaux"
scp -o BatchMode=yes mac2:/tmp/run_lookat_ft_ext.log results/logs/phase6_camera/run_lookat_ft_ext.log 2>/dev/null
# concatène les deux logs fine-tune pour la frise (15k + prolongation)
cat results/logs/phase6_camera/run_lookat_ft.log results/logs/phase6_camera/run_lookat_ft_ext.log \
  > results/logs/phase6_camera/run_lookat_ft_all.log 2>/dev/null
$PY experiments/can/59_plot_finetune_vs_run31.py 2>&1 | tail -2
$PY experiments/can/40_plot_model.py "$RUN" "results/logs/phase6_camera/run_lookat_ft*.log" \
  --title "Fine-tune look-at PROLONGÉ (30k) — 4 panneaux" --out "$RUN/full_curves.png" 2>&1 | tail -1
echo "[61] DONE — rollouts fine-tune prolongé :"; cat "$RUN/rollouts_100.csv"
