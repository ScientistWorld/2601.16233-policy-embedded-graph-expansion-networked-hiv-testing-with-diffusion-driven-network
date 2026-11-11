#!/bin/bash
# Visible train-slice evaluation.
#
# Scores every method on disk (checkpoints/<name>/policy.json) on the visible
# validation episodes in data/validation_episodes.json through the same
# interactive protocol as the held-out test, and writes
# scoring/scores_train.json. Runs with no arguments.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores_train.json "$@"
