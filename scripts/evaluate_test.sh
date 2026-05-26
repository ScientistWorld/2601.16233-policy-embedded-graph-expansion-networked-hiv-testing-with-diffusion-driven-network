#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh's artifact contract but writes
# scoring/scores_test.json and scores the disjoint held-out folds. If
# scoring/scores_train.json exists, the test scorer enforces the same method
# entries so train/test gaps are comparable.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  python eval/test/evaluate_slice.py --artifacts "$@" --output scoring/scores_test.json
else
  python eval/test/evaluate_slice.py --output scoring/scores_test.json
fi
