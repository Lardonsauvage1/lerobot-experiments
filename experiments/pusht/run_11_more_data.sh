#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : 100 épisodes ====="
python -u experiments/10_transformer.py --layers 2 --episodes 100

echo "===== RUN 2/2 : 206 épisodes (tout) ====="
python -u experiments/10_transformer.py --layers 2 --episodes 206

echo "===== TERMINÉ ====="
