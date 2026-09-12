#!/usr/bin/env bash
set -u
cd ~/lerobot-experiments
D=experiments/can
echo "[joint-all $(date '+%m-%d %H:%M')] === A (notre config) ==="; bash $D/50_joint_A_ourconfig.sh
echo "[joint-all $(date '+%m-%d %H:%M')] === CD A ==="; bash $D/52_joint_cooldown.sh joint_agent_ourconfig
echo "[joint-all $(date '+%m-%d %H:%M')] === B (bon config) ==="; bash $D/51_joint_B_goodconfig.sh
echo "[joint-all $(date '+%m-%d %H:%M')] === CD B ==="; bash $D/52_joint_cooldown.sh joint_agent_goodconfig
echo "[joint-all $(date '+%m-%d %H:%M')] === TOUT FINI (joint A/B) ==="
