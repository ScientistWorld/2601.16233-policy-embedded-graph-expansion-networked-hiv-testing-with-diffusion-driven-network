#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# No external assets are required for this scaled reproduction. The script
# writes all portable data into workspace-local tracked paths.

set -e

cd /home/user

# The ICPSR HIV network used by the paper requires a user agreement, so the
# portable gym uses a small procedural forest substitute generated from tracked
# code. No external data or model downloads are required.
python data/sapog_data.py --output data/synthetic_hiv_forest.json
