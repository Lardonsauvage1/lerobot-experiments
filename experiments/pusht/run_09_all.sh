#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/4 : chunk=1 (baseline) ====="
python -u experiments/09_action_chunking.py --chunk 1

echo "===== RUN 2/4 : chunk=5 ====="
python -u experiments/09_action_chunking.py --chunk 5

echo "===== RUN 3/4 : chunk=10 ====="
python -u experiments/09_action_chunking.py --chunk 10

echo "===== RUN 4/4 : chunk=20 ====="
python -u experiments/09_action_chunking.py --chunk 20

echo "===== TERMINÉ ====="
