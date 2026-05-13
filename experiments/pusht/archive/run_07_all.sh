#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : Sans filtre (baseline) ====="
python -u experiments/07_data_filtering.py

echo "===== RUN 2/2 : Avec filtre (reward > 0) ====="
python -u experiments/07_data_filtering.py --filter

echo "===== TERMINÉ ====="
echo "Comparer : python -c \"from src.tracker import compare_all; compare_all()\""
