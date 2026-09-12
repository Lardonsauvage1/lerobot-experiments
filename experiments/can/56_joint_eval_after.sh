#!/usr/bin/env bash
# Attend la fin de l'entraînement joint A/B puis lance les 2 évals 500 rollouts SUR gb10. 100% gb10.
set -u
cd ~/lerobot-experiments
echo "[eval-after $(date '+%H:%M')] attente fin entraînement joint..."
while ! grep -q "TOUT FINI (joint" /tmp/joint_all.log 2>/dev/null; do sleep 60; done
echo "[eval-after $(date '+%H:%M')] entraînement fini -> éval A (notre config)"
bash experiments/can/55_eval_joint_gb10.sh joint_agent_ourconfig 500 005000
echo "[eval-after $(date '+%H:%M')] éval B (bon config)"
bash experiments/can/55_eval_joint_gb10.sh joint_agent_goodconfig 500 005000
echo "[eval-after $(date '+%H:%M')] === EVALS JOINT FINIES ==="
