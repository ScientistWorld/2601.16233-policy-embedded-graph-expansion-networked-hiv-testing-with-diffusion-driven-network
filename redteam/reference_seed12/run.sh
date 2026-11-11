#!/bin/bash
set -eu
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python redteam/audit.py run reference_seed12
