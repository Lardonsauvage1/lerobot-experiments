#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/3 : CNN + MLP (baseline, 1 frame) ====="
python experiments/04_rnn_vs_mlp.py --model mlp --seq-len 1

echo "===== RUN 2/3 : CNN + RNN, 5 frames ====="
python experiments/04_rnn_vs_mlp.py --model rnn --seq-len 5

echo "===== RUN 3/3 : CNN + RNN, 10 frames ====="
python experiments/04_rnn_vs_mlp.py --model rnn --seq-len 10

echo ""
echo "===== TERMINÉ ====="
