#!/bin/bash
# Train the paper's method (PEGE + DDB, k=24, d=3, mean aggregation) on the
# public training forest and register it as checkpoints/pege_ddb/policy.json,
# then score it (with any other registered methods) on the visible slice.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python method/pege_ddb.py train --out checkpoints/pege_ddb --k "${PEGE_K:-24}" --depth "${PEGE_DEPTH:-3}" \
  --grid "${GITTINS_GRID:-256}" --aggregate "${PEGE_AGGREGATE:-mean}" --seed "${PEGE_SEED:-11}"
bash scripts/evaluate.sh
