#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : RNN seq=5 ====="
python -u experiments/06_pusht_rnn.py --seq-len 5

echo "===== RUN 2/2 : RNN seq=10 ====="
python -u experiments/06_pusht_rnn.py --seq-len 10

echo "===== TERMINÉ ====="
