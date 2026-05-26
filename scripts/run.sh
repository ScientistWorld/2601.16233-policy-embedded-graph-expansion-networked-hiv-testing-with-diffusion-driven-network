#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

# The DDB checkpoint schema changed to include the paper's linear SVM
# classifier comparator, so retrain fold models rather than resuming stale
# pickles from the previous smoke run.
rm -rf checkpoints/pege_ddb

# Run the corrected method at the paper's k=24, d=3, 5-fold protocol on the
# portable synthetic substitute. The Gittins grid is kept at 64 for compute.
PEGE_K="${PEGE_K:-24}" PEGE_DEPTH="${PEGE_DEPTH:-3}" FOLDS="${FOLDS:-5}" GITTINS_GRID="${GITTINS_GRID:-64}" bash scripts/method.sh
python validate.py --compare
