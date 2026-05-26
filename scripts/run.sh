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

# Validation-only job for the completed 5-fold PEGE+DDB artifact. The full
# end-to-end reproduction remains in scripts/reproduce.sh.
bash scripts/evaluate.sh checkpoints/pege_ddb/results.json
python validate.py --compare
