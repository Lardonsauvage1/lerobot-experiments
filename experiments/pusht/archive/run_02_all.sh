#!/bin/bash
# Lance les 5 runs de l'expérience 02

set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/5 : caméra côté + joints ====="
python experiments/02_mlp_cnn.py --cameras image

echo "===== RUN 2/5 : caméra haut + joints ====="
python experiments/02_mlp_cnn.py --cameras image2

echo "===== RUN 3/5 : caméra pince + joints ====="
python experiments/02_mlp_cnn.py --cameras hand_image

echo "===== RUN 4/5 : 3 caméras + joints ====="
python experiments/02_mlp_cnn.py --cameras image image2 hand_image

echo "===== RUN 5/5 : caméra côté sans joints ====="
python experiments/02_mlp_cnn.py --cameras image --no-joints

echo ""
echo "===== TOUS LES RUNS TERMINÉS ====="
echo "Pour comparer : python -c \"from src.tracker import compare_runs; compare_runs('02_mlp_cnn')\""
