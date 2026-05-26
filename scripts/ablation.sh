#!/bin/bash
# Run PEGE+DDB sample-count and aggregation ablations.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/sapog_data.py --output data/synthetic_hiv_forest.json
python method/pege_ablation.py \
  --data data/synthetic_hiv_forest.json \
  --output-dir checkpoints/pege_ablation \
  --folds "${FOLDS:-5}" \
  --depth "${PEGE_DEPTH:-3}" \
  --gittins-grid "${GITTINS_GRID:-64}" \
  --gamma 0.99

bash scripts/evaluate.sh \
  checkpoints/pege_ddb/results.json \
  checkpoints/pege_ablation/k_sweep.json \
  checkpoints/pege_ablation/aggregation.json
