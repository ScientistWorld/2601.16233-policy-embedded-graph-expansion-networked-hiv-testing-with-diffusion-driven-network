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

export FOLDS="${FOLDS:-5}"
export PEGE_K="${PEGE_K:-24}"
export PEGE_DEPTH="${PEGE_DEPTH:-3}"

bash scripts/reproduce.sh
python validate.py --compare
