#!/bin/bash
# ÉVAL des nouveaux checkpoints du joint poussé 52k->80k (n=50, kp=50, tous les 2k).
# Attend (a) le push joint FINI (80000/80000 sur mac2) et (b) le Principal LIBRE (toutes les
# évals run31-jumeau + fine-tune prolongé terminées). Transfère depuis mac2, évalue, régénère
# le 4-panneaux joint (log complet 0->80k). En FICHIER.
# Lancer : nohup bash experiments/can/63_eval_push_joint.sh > /tmp/push_joint_eval.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP="${MAC2_HOST:-mac2}"
RDJ=results/runs/can/joint_r34_bigunet

echo "[63] $(date '+%H:%M') attente fin du push joint (80000/80000)..."
while true; do
  d=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 \
    "pgrep -f '50_train_valloss.*joint_r34' >/dev/null && echo RUN || (grep -aoE '80000/80000' /tmp/run_joint_ext.log 2>/dev/null | tail -1)" 2>/dev/null)
  [ "$d" = "80000/80000" ] && break
  sleep 180
done
echo "[63] $(date '+%H:%M') push fini. Attente Principal libre (toutes évals terminées)..."
while pgrep -f "32_eval_dense_birdview|37_eval_joint_birdview|13_eval_camshift" >/dev/null; do sleep 120; done
echo "[63] $(date '+%H:%M') Principal libre -> transfert + éval n=50 des nouveaux checkpoints joint."

for s in 052000 054000 056000 058000 060000 062000 064000 066000 068000 070000 072000 074000 076000 078000 080000; do
  CK=$RDJ/checkpoints/$s/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RDJ/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RDJ" --steps "${s#0}" --n 50 --kp 50 --out "$RDJ/rollouts_50.csv"
done

# --- ÉVAL EMA : mêmes checkpoints 52-80k, poids EMA (dossier _ema) ---
RDE=results/runs/can/joint_r34_bigunet_ema
echo "[63] $(date '+%H:%M') éval EMA 52-80k (poids moyennés)..."
for s in 052000 054000 056000 058000 060000 062000 064000 066000 068000 070000 072000 074000 076000 078000 080000; do
  CK=$RDE/checkpoints/$s/pretrained_model
  if [ ! -s "$CK/model.safetensors" ]; then
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RDE/checkpoints/$s/pretrained_model" | tar xzf - -C . 2>/dev/null
  fi
  [ -s "$CK/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RDE" --steps "${s#0}" --n 50 --kp 50 --out "$RDE/rollouts_50.csv"
done
echo "[63] comparaison BRUT vs EMA :"
$PY experiments/can/64_plot_raw_vs_ema.py 2>&1 | tail -2

echo "[63] éval finie -> régénération 4-panneaux joint (log complet 0->80k)"
scp -o BatchMode=yes mac2:/tmp/run_joint_ext.log results/logs/can/run_joint_ext.log 2>/dev/null
cat results/logs/can/run_joint_full.log results/logs/can/run_joint_ext.log \
  > results/logs/can/run_joint_full2.log 2>/dev/null
mv "$RDJ/rollouts_500.csv" /tmp/j500_b.bak 2>/dev/null
$PY experiments/can/40_plot_model.py "$RDJ" "results/logs/can/run_joint_full2.log" \
  --title "JOINT 61M (moteur, 0->80k LR constant) — 4 panneaux" --out "$RDJ/full_curves_complet.png" 2>&1 | tail -1
mv /tmp/j500_b.bak "$RDJ/rollouts_500.csv" 2>/dev/null
echo "[63] DONE — convergence joint BRUT 0->80k :"; cut -d, -f1,4 "$RDJ/rollouts_50.csv"
echo "[63] convergence joint EMA 50-80k :"; cut -d, -f1,4 "$RDE/rollouts_50.csv" 2>/dev/null
