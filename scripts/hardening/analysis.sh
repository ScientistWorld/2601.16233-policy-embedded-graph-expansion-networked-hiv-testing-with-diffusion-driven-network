#!/bin/bash
# Hardening audit job: reference + baselines from final code, train/test
# scores, shortcut probes, honest better/weaker variants and seed spread on
# the hidden test.  Every stage is skipped when its output already exists, so
# a preempted job resumes where it stopped.
set -u
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=notes/hardening
mkdir -p $OUT/scores $OUT/details $OUT/logs tmp/hv
log() { echo "[$(date +%H:%M:%S)] $*"; }
nproc_avail=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
log "cpus available: $nproc_avail"

# ---- 1. reference method + baselines (from final code)
if [ ! -f checkpoints/pege_ddb/ddb.pkl ]; then
  log "train reference PEGE+DDB"
  python method/pege_ddb.py train --out checkpoints/pege_ddb --seed 11 || exit 1
fi
if [ ! -f checkpoints/greedy_classifier/model.pkl ]; then
  log "train baselines"
  python baseline/policies.py train --out-root checkpoints || exit 1
fi

# ---- 2. smoke test of every numpy path on the visible slice (fails fast)
if [ ! -f $OUT/scores/smoke_train.json ]; then
  log "smoke"
  rm -rf tmp/hv/smoke && python method/pege_ddb.py variant --base checkpoints/pege_ddb --out tmp/hv/smoke/pege_smoke --k 2 --depth 2 --grid 64 || exit 1
  python eval/train/evaluate.py --methods-root tmp/hv/smoke --output $OUT/scores/smoke_train.json || exit 1
  grep -q '"ok"' $OUT/scores/smoke_train.json || { cat $OUT/scores/smoke_train.json; ls logs/eval_train/pege_smoke | head; tail -50 logs/eval_train/pege_smoke/episode000.log; exit 1; }
fi

# ---- 3. official train / test scores (reference + baselines only)
if [ ! -f $OUT/details/train_reference.json ]; then
  log "evaluate_train"
  bash scripts/evaluate_train.sh --details $OUT/details/train_reference.json || exit 1
fi
if [ ! -f $OUT/details/test_reference.json ]; then
  log "evaluate_test"
  bash scripts/evaluate_test.sh --details $OUT/details/test_reference.json || exit 1
fi
cat scoring/scores_test.json

# ---- 4. shortcut probes (cheap, stdlib)
if [ ! -f $OUT/scores/probes_test.json ]; then
  log "probes"
  P=tmp/hv/probes; rm -rf $P; mkdir -p $P
  for m in random_control self_report invalid_action crash_midway hang hidden_file_read; do
    mkdir -p $P/$m
    echo "{\"command\": [\"python\", \"scripts/hardening/probe_policies.py\", \"$m\"]}" > $P/$m/policy.json
  done
  mkdir -p $P/forged_manifest
  echo '{"command": ["python", "scripts/hardening/probe_policies.py", "forged_manifest"], "gamma": 1.05, "budget": 310, "budget_fraction": 2.0, "experiment": "shortcut_unscored", "data": "/tmp/fake.json", "results": {"auc": 1e9}}' > $P/forged_manifest/policy.json
  mkdir -p $P/no_manifest
  echo '{"experiments":{"synthetic_main":{"results":{"random_control":{"auc":1e6,"reward_25pct":1e6,"reward_50pct":1e6,"reward_75pct":1e6}}}}}' > $OUT/scores/probes_test.json
  SAPOG_STEP_TIMEOUT=20 python eval/test/evaluate.py --methods-root $P --output $OUT/scores/probes_test.json --details $OUT/details/probes_test.json
  # interrupted evaluation must leave worst values, not stale scores
  echo '{"experiments":{"synthetic_main":{"results":{"random_control":{"auc":1e6}}}}}' > $OUT/scores/probes_interrupted_test.json
  timeout -s KILL 4 python eval/test/evaluate.py --methods-root $P --output $OUT/scores/probes_interrupted_test.json
  cat $OUT/scores/probes_interrupted_test.json
fi

# ---- 5. FOG (privileged, diagnostic only) on validation
if [ ! -f $OUT/scores/fog_validation.json ]; then
  log "fog diagnostic"
  python method/fog_diagnostic.py --model-dir checkpoints/pege_ddb --out $OUT/scores/fog_validation.json
fi

# ---- 6. honest variants on the hidden test (each its own methods root)
run_variant() {  # name, then 'variant' or 'train' args
  local name=$1; shift
  local kind=$1; shift
  if [ -f $OUT/details/${name}_test.json ]; then return 0; fi
  log "variant $name"
  local dir=tmp/hv/$name/$name
  if [ ! -f $dir/policy.json ]; then
    rm -rf tmp/hv/$name
    if [ "$kind" = variant ]; then
      python method/pege_ddb.py variant --base checkpoints/pege_ddb --out $dir "$@" || return 1
    else
      python method/pege_ddb.py train --out $dir "$@" || return 1
    fi
  fi
  python eval/test/evaluate.py --methods-root tmp/hv/$name --output $OUT/scores/${name}_test.json --details $OUT/details/${name}_test.json
  python scripts/hardening/summarize.py > $OUT/summary.txt 2>&1
}
run_variant no_expansion variant --depth 0
run_variant k2 variant --k 2
run_variant theta_parent_features variant --theta parent_features
run_variant seed12 train --seed 12
run_variant seed13 train --seed 13
run_variant mean_plus_var variant --aggregate mean_plus_var
run_variant k48 variant --k 48
run_variant mode variant --aggregate mode
run_variant mean_minus_var variant --aggregate mean_minus_var

python scripts/hardening/summarize.py > $OUT/summary.txt 2>&1
cat $OUT/summary.txt
log "done"
