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

# Re-run the corrected PEGE+DDB implementation with the paper's main PEGE
# sample/depth settings on the portable synthetic benchmark.
rm -rf checkpoints/pege_ddb
PEGE_K=24 PEGE_DEPTH=3 FOLDS=5 bash scripts/method.sh
python validate.py --compare
