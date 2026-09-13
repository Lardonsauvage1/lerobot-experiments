#!/bin/bash
cd $HOME/lerobot-experiments
source $HOME/roby_xpu_env.sh
source /opt/ros/jazzy/setup.bash
exec $HOME/ipex_test_venv/bin/python -m lerobot.scripts.lerobot_train \
     --config_path=$HOME/train_config_pince.json
