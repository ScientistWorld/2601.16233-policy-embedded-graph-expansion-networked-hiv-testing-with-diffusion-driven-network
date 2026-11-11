#!/bin/bash
# Hardening job: reference + baselines from final code, official train/test
# scores, shortcut probes, honest variants and seed spread (resumable).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/hardening/analysis.sh
python validate.py
