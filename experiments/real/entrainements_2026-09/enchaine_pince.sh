#!/bin/bash
# Attend la fin de la phase 1 (LR constant) puis lance le cooldown. Sans intervention.
set -u
CK=$HOME/lerobot-experiments/outputs/pince_const_20260909/checkpoints/036000/pretrained_model
LOG=$HOME/pince_enchainement.log
echo "$(date +%H:%M) attente de la phase 1 (36000 pas)..." >> $LOG
# on attend que le checkpoint final existe ET que l'entrainement se soit arrete
until [ -d "$CK" ]; do sleep 300; done
echo "$(date +%H:%M) checkpoint 036000 present" >> $LOG
while pgrep -f 'train_config_[p]ince.json' > /dev/null; do sleep 60; done
echo "$(date +%H:%M) phase 1 terminee -> lancement du cooldown" >> $LOG
cd $HOME/lerobot-experiments
source $HOME/roby_xpu_env.sh
source /opt/ros/jazzy/setup.bash
$HOME/ipex_test_venv/bin/python -m lerobot.scripts.lerobot_train \
    --config_path=$HOME/train_config_pince_cooldown.json \
    >> $HOME/pince_cooldown_20260909.log 2>&1
echo "$(date +%H:%M) COOLDOWN TERMINE" >> $LOG
