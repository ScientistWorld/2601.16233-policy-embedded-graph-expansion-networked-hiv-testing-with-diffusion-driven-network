#!/bin/bash
set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/sapog_data.py --output data/synthetic_hiv_forest.json

# Recompute or resume the main five-fold PEGE+DDB benchmark after method
# changes. The method runner writes each fold/policy as it finishes, so keeping
# the checkpoint directory lets preempted CPU jobs continue instead of starting
# over.
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

# Additional paper-aligned ablations for the next milestone: sample-count
# sweep and PEGE score aggregation variants from Figure 6.
bash scripts/ablation.sh
bash scripts/evaluate_train.sh checkpoints/pege_ddb/results.json
bash scripts/evaluate_test.sh checkpoints/pege_ddb/results.json
python validate.py --compare
