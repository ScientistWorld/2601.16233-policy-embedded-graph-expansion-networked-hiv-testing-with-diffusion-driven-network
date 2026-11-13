#!/bin/bash
# Held-out test evaluation (hidden slice).
#
# Scores every method on disk: each directory checkpoints/<name>/ with a
# policy.json is played interactively on held-out episodes generated inside
# eval/test/ and written to scoring/scores_test.json. Gamma, budget, episodes
# and metrics are fixed by eval/test/; a failed method gets the worst value
# in every metric. eval/test/ is made unreadable while policies run; a killed
# run can leave it locked, so its permissions are restored first. Runs with no
# arguments (extra flags are passed through, e.g. --details PATH for
# per-episode analysis).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
chmod u+rwx eval/test 2>/dev/null || true
python eval/test/evaluate.py --output scoring/scores_test.json "$@"
