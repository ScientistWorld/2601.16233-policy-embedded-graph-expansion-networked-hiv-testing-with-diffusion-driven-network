#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB (no external assets).
#
# The paper's ICPSR 22140 HIV network requires a signed ICPSR user agreement
# and cannot be fetched non-interactively, so this gym uses a synthetic
# referral-forest substitute. Its public part (data/train_forest.json and
# data/validation_episodes.json) is small, agent-authored and committed to
# git; held-out test episodes are generated inside eval/test/ at test time.
# There is nothing to fetch; data/downloads/ stays empty.
set -e
cd /home/user
mkdir -p data/downloads
for f in data/train_forest.json data/validation_episodes.json; do
  if [ ! -s "$f" ]; then
    echo "missing committed data file $f (restore it with git checkout -- $f)" >&2
    exit 1
  fi
done
echo "no external downloads required"
