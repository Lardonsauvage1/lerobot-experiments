#!/usr/bin/env bash
# Orchestrateur PERF (gb10) : train -> cooldown, en séquence, pour une résolution.
# Usage : bash experiments/real/44_perf_run_all.sh 128   (dans tmux 'perf')
set -u
RES=${1:-128}
cd ~/lerobot-experiments
echo "===== [PERF $RES] TRAIN $(date) ====="
bash experiments/real/42_train_b2_perf.sh $RES || { echo "TRAIN ECHEC"; exit 1; }
echo "===== [PERF $RES] COOLDOWN $(date) ====="
bash experiments/real/43_cooldown_perf_gb10.sh $RES || { echo "COOLDOWN ECHEC"; exit 1; }
echo "===== [PERF $RES] TOUT FINI $(date) ====="
