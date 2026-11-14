#!/bin/bash
# Reproduce the reference end to end: train PEGE + DDB (seed 11) and the
# paper's baselines on the public training forest, score them on the visible
# slice (scores.json, scores_train.json) and the held-out slice
# (scores_test.json), check the result against the recorded reference scores,
# and validate the workspace.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/reproduce.sh
python scripts/hardening/compare_reference.py || echo "WARNING: reproduction differs from the recorded reference scores"
python validate.py
