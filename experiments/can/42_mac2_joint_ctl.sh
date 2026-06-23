#!/bin/bash
# Contrôle du jumeau articulaire sur mac2 DEPUIS le Mac principal (via ssh).
# Arrêt PROPRE = reprise depuis checkpoints/last (perte ≤ save_freq=2000 steps, ~33 min au pire).
#
#   bash experiments/can/42_mac2_joint_ctl.sh status   # où en est-il
#   bash experiments/can/42_mac2_joint_ctl.sh stop      # arrêt propre (libère mac2)
#   bash experiments/can/42_mac2_joint_ctl.sh resume    # reprend depuis le dernier checkpoint

RUN="results/runs/can/joint_r34_bigunet"
PAT="50_train_valloss.*joint_r34"

case "${1:-status}" in
  status)
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments; \
      pgrep -f '$PAT' >/dev/null && echo 'EN COURS' || echo 'ARRÊTÉ'; \
      grep -ao '[0-9]*/40000 \[[0-9:]*<[0-9:]*' /tmp/run_joint.log | tail -1; \
      echo -n 'dernier checkpoint: '; ls $RUN/checkpoints/ | grep -E '^[0-9]+\$' | sort -n | tail -1"
    ;;
  stop)
    ssh -o BatchMode=yes mac2 "pkill -f '$PAT'; sleep 4; \
      pgrep -f '$PAT' >/dev/null && echo '⚠️ encore vivant' || echo '✓ ARRÊTÉ PROPREMENT'; \
      echo -n 'reprise possible depuis checkpoint: '; ls ~/lerobot-experiments/$RUN/checkpoints/ | grep -E '^[0-9]+\$' | sort -n | tail -1"
    echo "→ mac2 libéré. Relancer avec : bash experiments/can/42_mac2_joint_ctl.sh resume"
    ;;
  resume)
    ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && \
      CONST_LR=1 CONST_LR_VALUE=1e-4 nohup venv312/bin/python -u experiments/lift/50_train_valloss.py \
        --config_path=$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true \
        --output_dir=$RUN --steps=40000 --save_freq=2000 --log_freq=50 --eval_freq=0 \
        >> /tmp/run_joint.log 2>&1 & echo \"repris PID \$!\""
    sleep 6
    ssh -o BatchMode=yes mac2 "P=\$(pgrep -f '$PAT' | head -1); \
      if [ -n \"\$P\" ]; then nohup caffeinate -w \$P >/dev/null 2>&1 & echo \"✓ repris (PID \$P) + caffeinate\"; \
      grep -ao 'resume\|[0-9]*/40000' /tmp/run_joint.log | tail -2; else echo '⚠️ pas reparti, voir /tmp/run_joint.log'; fi"
    ;;
  *) echo "usage: $0 {status|stop|resume}" ;;
esac
