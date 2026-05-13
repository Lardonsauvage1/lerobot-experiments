#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : CNN from scratch + MLP ====="
python -u experiments/08_pretrained_cnn.py

echo "===== RUN 2/2 : ResNet18 pré-entraîné + MLP ====="
python -u experiments/08_pretrained_cnn.py --pretrained

echo "===== TERMINÉ ====="
