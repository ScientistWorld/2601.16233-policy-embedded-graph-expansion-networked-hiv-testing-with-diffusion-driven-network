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

# Run a resumable corrected PEGE+DDB job. This keeps the paper's procedure
# but uses a smaller sample count/fold count for the first corrected result;
# scripts/method.sh defaults remain at the paper's k=24, d=3 setting.
PEGE_K="${PEGE_K:-8}" PEGE_DEPTH="${PEGE_DEPTH:-3}" FOLDS="${FOLDS:-3}" GITTINS_GRID="${GITTINS_GRID:-64}" bash scripts/method.sh
python validate.py --compare
