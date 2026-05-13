#!/bin/bash
# Expérience 03 : impact de la représentation sin/cos

set -e
cd "$(dirname "$0")/.."

echo "===== RUN 1/2 : sin/cos sur joint 3 uniquement ====="
python experiments/03_sincos.py --sincos-joints 3

echo "===== RUN 2/2 : sin/cos sur tous les joints ====="
python experiments/03_sincos.py --sincos-joints 0 1 2 3 4 5

echo ""
echo "===== TERMINÉ ====="
echo "Comparer : python -c \"from src.tracker import compare_runs; compare_runs('03_sincos')\""
