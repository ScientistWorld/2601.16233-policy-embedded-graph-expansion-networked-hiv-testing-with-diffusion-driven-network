#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh's artifact contract but writes
# scoring/scores_train.json and scores only the visible anti-shortcut folds.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  python eval/train/evaluate_slice.py --artifacts "$@" --output scoring/scores_train.json
else
  python eval/train/evaluate_slice.py --output scoring/scores_train.json
fi
