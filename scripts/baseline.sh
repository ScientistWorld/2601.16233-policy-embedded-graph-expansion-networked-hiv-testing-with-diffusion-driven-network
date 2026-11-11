#!/bin/bash
# Register the paper's baselines (Random, Greedy Neighbor, Greedy Classifier)
# as checkpoints/<name>/policy.json and score them on the visible slice.
#
# Usage: bash scripts/baseline.sh [comma-separated kinds]
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python baseline/policies.py train --out-root checkpoints --kinds "${1:-random,greedy_neighbor,greedy_classifier}"
bash scripts/evaluate.sh
