#!/bin/bash
# Éval de l'extension joint 80k->100k : rollouts BRUTS (84-100k tous les 4k, comme le push)
# + 2 fenêtres rivière (e86=82-90k, e96=92-100k, 5 ckpts 2k-espacés, comme avant).
# Attend la fin du training 100k (mac2) ET Principal libre (76 fini). En FICHIER.
# Lancer : nohup caffeinate -i bash experiments/can/78_eval_river_100k.sh > /tmp/river_100k.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RDJ=results/runs/can/joint_r34_bigunet
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP="${MAC2_HOST:-mac2}"

echo "[78] $(date '+%H:%M') attente fin training joint 100k (mac2)..."
while true; do
  d=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*joint_r34' >/dev/null && echo RUN || (grep -aoE '20000/20000' /tmp/run_joint_100k.log 2>/dev/null | tail -1)" 2>/dev/null)
  [ "$d" = "20000/20000" ] && break
  [ -z "$d" ] && { echo "[78] mac2 injoignable, retry"; sleep 120; continue; }
  sleep 180
done
echo "[78] $(date '+%H:%M') training fini. Priorité au mini-CNN river (33) : attente sa fin..."
# le mini-CNN river (33) est prioritaire -> on attend qu'il soit lancé/fini avant de prendre Principal
while pgrep -f "33_river_minicnn|12_eval_parallel" >/dev/null; do sleep 60; done
echo "[78] $(date '+%H:%M') attente Principal libre (évals MPS)..."
while pgrep -f "37_eval_joint_birdview|32_eval_dense_birdview|47_eval_joint_ensemble|50_train_valloss|12_eval_parallel" >/dev/null; do sleep 60; done

echo "[78] transfert checkpoints 82-100k (tous les 2k) depuis mac2"
for s in 082000 084000 086000 088000 090000 092000 094000 096000 098000 100000; do
  CK=$RDJ/checkpoints/$s/pretrained_model
  [ -s "$CK/model.safetensors" ] && continue
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RDJ/checkpoints/$s/pretrained_model 2>/dev/null" | tar xzf - -C . 2>/dev/null
done

echo "[78] rollouts BRUTS 84-100k (n=50, tous les 4k)"
for s in 084000 088000 092000 096000 100000; do
  [ -s "$RDJ/checkpoints/$s/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RDJ" --steps "${s#0}" --n 50 --kp 50 --out "$RDJ/rollouts_50.csv"
done

echo "[78] fenêtres rivière e86 (82-90k) + e96 (92-100k)"
$PY experiments/can/65_swa_make.py --src "$RDJ" --steps 82000,84000,86000,88000,90000 --out results/runs/can/joint_swa_e86_82_90k 2>&1|tail -1
$PY experiments/can/65_swa_make.py --src "$RDJ" --steps 92000,94000,96000,98000,100000 --out results/runs/can/joint_swa_e96_92_100k 2>&1|tail -1
for w in e86_82_90k e96_92_100k; do
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir results/runs/can/joint_swa_$w --steps 0 --n 50 --kp 50 --out results/runs/can/joint_swa_$w/r.csv
done

echo "[78] régénération graphe rivière (jusqu'à 100k)"
$PY experiments/can/77_plot_river.py 2>&1 | tail -1
echo "[78] DONE — fond e86=$(cut -d,-f4 results/runs/can/joint_swa_e86_82_90k/r.csv 2>/dev/null|tail -1) e96=$(cut -d,-f4 results/runs/can/joint_swa_e96_92_100k/r.csv 2>/dev/null|tail -1)"
