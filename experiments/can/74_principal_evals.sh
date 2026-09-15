#!/bin/bash
# PIPELINE ÉVALS Principal (2 machines : le push joint tourne sur mac2 en parallèle).
# P1 matrice SWA -> P2 run31-jumeau 26k->2k -> P3 profil rivière -> P4 cos_mid -> P5 éval push (attend mac2).
# Tolérant aux erreurs. Lancer : nohup caffeinate -i bash experiments/can/74_principal_evals.sh > /tmp/evals.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP="${MAC2_HOST:-mac2}"
RDJ=results/runs/can/joint_r34_bigunet
log(){ echo "[74] $(date '+%H:%M') $*"; }

# ===== P1 : MATRICE SWA =====
log "P1 matrice SWA"
CKJ=results/runs/can/joint_swa_late10/checkpoints/000000/pretrained_model
$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CKJ" --no-ensemble --steps 0 --n 50 --kp 50 --out results/runs/can/joint_swa_late10/baseline.csv || log "FAIL joint SWA baseline"
$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CKJ" --m 0.01 --steps 0 --n 50 --kp 50 --out results/runs/can/joint_swa_late10/ensemble.csv || log "FAIL joint SWA ensemble"
$PY experiments/can/65_swa_make.py --src results/runs/can/run31_constLR --steps 32000,34000,36000,38000,40000 --out results/runs/can/run31jum_swa_const_late5 2>&1|tail -1
$PY experiments/can/65_swa_make.py --src results/runs/can/run31_constLR --steps 22000,24000,26000,28000,30000,32000,34000,36000,38000,40000 --out results/runs/can/run31jum_swa_const_late10 2>&1|tail -1
for n in const_late5 const_late10; do $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31jum_swa_$n --steps 0 --n 100 --out results/runs/can/run31jum_swa_$n/r.csv || log "FAIL $n"; done
for n in cos_late5 cos_late10 cos_mid; do $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31_swa_$n --steps 0 --n 100 --out results/runs/can/run31_swa_$n/r.csv || log "FAIL $n"; done
log "P1 DONE"

# ===== P2 : run31-jumeau 26k->2k =====
log "P2 run31-jumeau 26k->2k"
for s in 026000 024000 022000 020000 018000 016000 014000 012000 010000 008000 006000 004000 002000; do
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31_constLR --steps "${s#0}" --n 100 --out results/runs/can/run31_constLR/rollouts_100.csv || log "FAIL run31jum $s"
done
$PY experiments/can/40_plot_model.py results/runs/can/run31_constLR "results/logs/can/run31_constLR.log" --title "run31-jumeau LR constant — 4 panneaux" --out results/runs/can/run31_constLR/full_curves.png || log "FAIL plot run31jum"
log "P2 DONE"

# ===== P3 : PROFIL DE LA RIVIÈRE =====
log "P3 profil rivière (fonds SWA par fenêtre)"
for w in W1_12_20k W2_22_30k W3_32_40k W4_42_50k; do
  $PY -u experiments/can/37_eval_joint_birdview.py --run-dir results/runs/can/joint_swa_$w --steps 0 --n 50 --kp 50 --out results/runs/can/joint_swa_$w/r.csv || log "FAIL river $w"
done
log "P3 DONE — profil rivière :"; for w in W1_12_20k W2_22_30k W3_32_40k W4_42_50k; do echo "  $w fond = $(cut -d, -f4 results/runs/can/joint_swa_$w/r.csv 2>/dev/null|tail -1)"; done

# ===== P4 : éval push (attend mac2) =====
log "P4 attente fin du push mac2 (80000/80000)..."
while true; do
  d=$(ssh -o BatchMode=yes -o ConnectTimeout=15 mac2 "pgrep -f '50_train_valloss.*joint_r34' >/dev/null && echo RUN || (grep -aoE '80000/80000' /tmp/run_joint_ext.log 2>/dev/null|tail -1)" 2>/dev/null)
  [ "$d" = "80000/80000" ] && break
  [ -z "$d" ] && { log "mac2 injoignable, P4 reporté"; break; }
  sleep 180
done
if ssh -o BatchMode=yes mac2 "grep -aqE '80000/80000' /tmp/run_joint_ext.log" 2>/dev/null; then
  log "P4 éval push brut + EMA (n=50, tous les 4k)"
  for s in 054000 058000 062000 066000 070000 074000 078000 080000; do
    "$TS" ping -c 2 "$IP" >/dev/null 2>&1
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $RDJ/checkpoints/$s/pretrained_model ${RDJ}_ema/checkpoints/$s/pretrained_model 2>/dev/null" | tar xzf - -C . 2>/dev/null
    [ -s "$RDJ/checkpoints/$s/pretrained_model/model.safetensors" ] && $PY -u experiments/can/37_eval_joint_birdview.py --run-dir $RDJ --steps "${s#0}" --n 50 --kp 50 --out $RDJ/rollouts_50.csv
    [ -s "${RDJ}_ema/checkpoints/$s/pretrained_model/model.safetensors" ] && $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "${RDJ}_ema" --steps "${s#0}" --n 50 --kp 50 --out "${RDJ}_ema/rollouts_50.csv"
  done
  scp -o BatchMode=yes mac2:/tmp/run_joint_ext.log results/logs/can/run_joint_ext.log 2>/dev/null
  cat results/logs/can/run_joint_full.log results/logs/can/run_joint_ext.log > results/logs/can/run_joint_full2.log 2>/dev/null
  mv "$RDJ/rollouts_500.csv" /tmp/j5.bak 2>/dev/null
  $PY experiments/can/40_plot_model.py "$RDJ" "results/logs/can/run_joint_full2.log" --title "JOINT 0->80k LR constant — 4 panneaux" --out "$RDJ/full_curves_complet.png"
  mv /tmp/j5.bak "$RDJ/rollouts_500.csv" 2>/dev/null
fi
log "P4 DONE — TOUT FINI"
