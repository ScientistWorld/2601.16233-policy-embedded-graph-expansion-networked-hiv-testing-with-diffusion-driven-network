#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/sapog_data.py --output data/synthetic_hiv_forest.json
python method/pege_ddb.py \
  --data data/synthetic_hiv_forest.json \
  --output-dir checkpoints/pege_ddb \
  --experiment synthetic_main \
  --policies pege_ddb,greedy_classifier,greedy_neighbor,random,fog \
  --folds "${FOLDS:-5}" \
  --k "${PEGE_K:-12}" \
  --depth "${PEGE_DEPTH:-3}" \
  --gamma 0.99

bash scripts/evaluate.sh checkpoints/pege_ddb/results.json
