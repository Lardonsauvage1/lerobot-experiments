#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : Transformer 1 couche, chunk=20 ====="
python -u experiments/10_transformer.py --layers 1

echo "===== RUN 2/2 : Transformer 2 couches, chunk=20 ====="
python -u experiments/10_transformer.py --layers 2

echo "===== TERMINÉ ====="
