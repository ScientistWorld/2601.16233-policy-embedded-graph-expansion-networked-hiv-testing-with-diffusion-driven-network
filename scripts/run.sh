#!/bin/bash
set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/sapog_data.py --output data/synthetic_hiv_forest.json

# Recompute the main five-fold PEGE+DDB benchmark after method changes. The
# checkpoint directory is generated output and is intentionally gitignored.
rm -rf checkpoints/pege_ddb
python method/pege_ddb.py \
  --data data/synthetic_hiv_forest.json \
  --output-dir checkpoints/pege_ddb \
  --experiment synthetic_main \
  --policies pege_ddb,greedy_classifier,greedy_neighbor,random,fog \
  --folds 5 \
  --k 24 \
  --depth 3 \
  --gittins-grid 64 \
  --pege-aggregate mean \
  --gamma 0.99

bash scripts/evaluate.sh checkpoints/pege_ddb/results.json
bash scripts/evaluate_train.sh checkpoints/pege_ddb/results.json
bash scripts/evaluate_test.sh checkpoints/pege_ddb/results.json
python validate.py --compare
