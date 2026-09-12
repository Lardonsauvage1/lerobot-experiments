#!/bin/bash
# Attend la fin de la conversion du dataset joint, puis lance le jumeau articulaire SUR mac2 :
# resync code (patch CONST_LR + script 38) -> transfert dataset joint -> lancement -> caffeinate.
# Lancer : nohup bash experiments/can/39_launch_joint_on_mac2.sh > results/logs/can/run_39_launch_joint_mac2.log 2>&1 &

cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1

echo "[39] $(date '+%H:%M:%S') attente fin conversion dataset joint"
until grep -aq "Dataset créé" /tmp/convert_joint.log 2>/dev/null; do sleep 30; done
echo "[39] $(date '+%H:%M:%S') dataset joint prêt"

tailscale ping -c 2 mac2 >/dev/null 2>&1 || true   # réveille le lien tailscale (cold start)

echo "[39] resync code -> mac2"
tar czf - experiments src requirements.txt | ssh -o BatchMode=yes mac2 'tar xzf - -C ~/lerobot-experiments' && echo "[39] code OK"

echo "[39] transfert dataset joint -> mac2"
tar czf - -C data_cache lerobot_can_ph_joint_birdview | ssh -o BatchMode=yes mac2 'mkdir -p ~/lerobot-experiments/data_cache && tar xzf - -C ~/lerobot-experiments/data_cache' && echo "[39] dataset OK"

echo "[39] lancement training sur mac2"
ssh -o BatchMode=yes mac2 'cd ~/lerobot-experiments && nohup bash experiments/can/38_train_joint_mac2.sh > /tmp/run_joint.log 2>&1 & echo "  mac2 PID $!"'
sleep 5
ssh -o BatchMode=yes mac2 'P=$(pgrep -f "50_train_valloss.*joint_r34" | head -1); if [ -n "$P" ]; then nohup caffeinate -w "$P" >/dev/null 2>&1 & echo "  caffeinate sur $P"; fi'
echo "[39] $(date '+%H:%M:%S') JOINT LANCÉ SUR MAC2 (log mac2: /tmp/run_joint.log)"
