#!/bin/bash
# Standard evaluation: score every method under checkpoints/ on the visible
# validation episodes (same protocol and metrics as the held-out test) and
# write scoring/scores.json. Uses only code in eval/ (never method/).
#
# Usage: bash scripts/evaluate.sh [--methods-root DIR]
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json "$@"
