#!/bin/bash
# PIPELINE MAÎTRE Principal-only (mac2 abandonné). Séquentiel (1 MPS à la fois).
# Phase 1 matrice SWA (rapide, gros ROI) -> Phase 2 run31-jumeau 26k->2k -> Phase 3 push joint 50k->80k+EMA
# -> Phase 4 push eval allégé (brut+EMA). Tolérant aux erreurs (continue si une étape plante).
# Lancer : nohup bash experiments/can/71_principal_master.sh > /tmp/master.log 2>&1 &
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
RDJ=results/runs/can/joint_r34_bigunet
log(){ echo "[71] $(date '+%H:%M') $*"; }

# ===================== PHASE 1 : MATRICE SWA =====================
log "PHASE 1 — matrice SWA"
# 1a) joint SWA late10 : baseline (sans) + temporal ensembling
CKJ=results/runs/can/joint_swa_late10/checkpoints/000000/pretrained_model
$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CKJ" --no-ensemble --steps 0 --n 50 --kp 50 \
  --out results/runs/can/joint_swa_late10/baseline.csv || log "FAIL 1a baseline"
$PY -u experiments/can/47_eval_joint_ensemble.py --ckpt "$CKJ" --m 0.01 --steps 0 --n 50 --kp 50 \
  --out results/runs/can/joint_swa_late10/ensemble.csv || log "FAIL 1a ensemble"
# 1b) SWA constant cartésien (cellule manquante) : build + eval
$PY experiments/can/65_swa_make.py --src results/runs/can/run31_constLR \
  --steps 32000,34000,36000,38000,40000 --out results/runs/can/run31jum_swa_const_late5 || log "FAIL 1b make late5"
$PY experiments/can/65_swa_make.py --src results/runs/can/run31_constLR \
  --steps 22000,24000,26000,28000,30000,32000,34000,36000,38000,40000 --out results/runs/can/run31jum_swa_const_late10 || log "FAIL 1b make late10"
for n in const_late5 const_late10; do
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31jum_swa_$n --steps 0 --n 100 \
    --out results/runs/can/run31jum_swa_$n/r.csv || log "FAIL 1b eval $n"
done
# 1c) SWA cosine (modèles déjà construits)
for n in cos_late5 cos_late10; do
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31_swa_$n --steps 0 --n 100 \
    --out results/runs/can/run31_swa_$n/r.csv || log "FAIL 1c eval $n"
done
log "PHASE 1 DONE — matrice SWA"

# ===================== PHASE 2 : run31-jumeau 26k->2k =====================
log "PHASE 2 — run31-jumeau completion (26k->2k, n=100)"
for s in 026000 024000 022000 020000 018000 016000 014000 012000 010000 008000 006000 004000 002000; do
  $PY -u experiments/can/32_eval_dense_birdview.py --run-dir results/runs/can/run31_constLR --steps "${s#0}" --n 100 \
    --out results/runs/can/run31_constLR/rollouts_100.csv || log "FAIL 2 $s"
done
$PY experiments/can/40_plot_model.py results/runs/can/run31_constLR "results/logs/can/run31_constLR.log" \
  --title "run31-jumeau LR constant 1e-4 (cartesien) — 4 panneaux" \
  --out results/runs/can/run31_constLR/full_curves.png || log "FAIL 2 plot"
log "PHASE 2 DONE — run31-jumeau"

# ===================== PHASE 3 : push joint 50k->80k + EMA =====================
log "PHASE 3 — push joint 50k->80k + EMA (training)"
rm -rf "$RDJ/checkpoints/last"; ln -s 050000 "$RDJ/checkpoints/last"
CONST_LR=1 CONST_LR_VALUE=1e-4 EMA=1 EMA_DECAY=0.9999 $PY -u experiments/lift/50_train_valloss.py \
  --config_path=$RDJ/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --output_dir=$RDJ \
  --steps=80000 --save_freq=2000 --log_freq=50 --eval_freq=0 >> /tmp/run_joint_ext.log 2>&1 || log "FAIL 3 training"
log "PHASE 3 DONE — push training"

# ===================== PHASE 4 : push eval allégé (n=50 tous les 4k) brut + EMA =====================
log "PHASE 4 — push eval (brut + EMA, n=50, tous les 4k)"
for s in 054000 058000 062000 066000 070000 074000 078000 080000; do
  [ -s "$RDJ/checkpoints/$s/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir $RDJ --steps "${s#0}" --n 50 --kp 50 --out $RDJ/rollouts_50.csv || log "FAIL 4 brut $s"
  CKE="${RDJ}_ema/checkpoints/$s/pretrained_model"
  [ -s "$CKE/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "${RDJ}_ema" --steps "${s#0}" --n 50 --kp 50 --out "${RDJ}_ema/rollouts_50.csv" || log "FAIL 4 ema $s"
done
# graphes joint 0->80k (le push tourne en LOCAL -> log local /tmp/run_joint_ext.log)
cp /tmp/run_joint_ext.log results/logs/can/run_joint_ext.log 2>/dev/null
cat results/logs/can/run_joint_full.log /tmp/run_joint_ext.log > results/logs/can/run_joint_full2.log 2>/dev/null
mv "$RDJ/rollouts_500.csv" /tmp/j5.bak 2>/dev/null
$PY experiments/can/40_plot_model.py "$RDJ" "results/logs/can/run_joint_full2.log" \
  --title "JOINT 61M (moteur, 0->80k LR constant) — 4 panneaux" --out "$RDJ/full_curves_complet.png" || log "FAIL 4 plot"
mv /tmp/j5.bak "$RDJ/rollouts_500.csv" 2>/dev/null
log "PHASE 4 DONE — TOUT FINI"
