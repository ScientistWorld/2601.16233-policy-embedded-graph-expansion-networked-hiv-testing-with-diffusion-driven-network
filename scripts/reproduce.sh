#!/bin/bash
# Reproduce the reference end to end: train PEGE + DDB and the baselines on
# the public training forest, then score every registered method on the
# visible slice (scores.json, scores_train.json) and the held-out slice
# (scores_test.json).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python method/pege_ddb.py train --out checkpoints/pege_ddb --seed 11
python baseline/policies.py train --out-root checkpoints
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
