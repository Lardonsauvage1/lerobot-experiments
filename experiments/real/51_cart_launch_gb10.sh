#!/usr/bin/env bash
cd ~/lerobot-experiments
EPS=$(cat /tmp/eps.txt) bash experiments/real/50_cart_combined_run_all.sh
