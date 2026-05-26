#!/bin/bash
set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
