#!/bin/bash
# Hardening audit job (round 2): fixed DDB reference + baselines from final
# code, official train/test scores, hand-written-rule probes, shortcut probes,
# honest better/weaker variants and seed spread on the hidden test.  Every
# stage is skipped when its output already exists, so a preempted job resumes
# where it stopped.  Round-1 outputs (80 episodes, previous DDB) are kept in
# notes/hardening/round1/ for before/after comparisons.
set -u
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=notes/hardening
mkdir -p $OUT/scores $OUT/details tmp/hv2
log() { echo "[$(date +%H:%M:%S)] $*"; }
heval() {  # hidden evaluator; restore eval/test permissions around it
  chmod u+rwx eval/test 2>/dev/null
  python eval/test/evaluate.py "$@"
  local rc=$?
  chmod u+rwx eval/test 2>/dev/null
  return $rc
}
log "cpus available: $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"

# ---- 1. reference method + baselines from the final code
if [ ! -f tmp/hv2/reference_trained ]; then
  if [ -f checkpoints/pege_ddb/ddb.pkl ] && [ ! -f tmp/hv2/previous_ddb.pkl ]; then
    cp checkpoints/pege_ddb/ddb.pkl tmp/hv2/previous_ddb.pkl
  fi
  rm -rf checkpoints/pege_ddb
  log "train reference PEGE+DDB"
  python method/pege_ddb.py train --out checkpoints/pege_ddb --seed 11 || exit 1
  log "train baselines"
  python baseline/policies.py train --out-root checkpoints || exit 1
  touch tmp/hv2/reference_trained
fi

# ---- 2. DDB diagnostics (previous vs fixed expansion model)
if [ ! -f $OUT/scores/ddb_diagnostics.json ]; then
  log "ddb diagnostics"
  python scripts/hardening/ddb_diagnostics.py --old tmp/hv2/previous_ddb.pkl --out $OUT/scores/ddb_diagnostics.json || exit 1
fi

# ---- 3. smoke test of every numpy path on the visible slice (fails fast)
if [ ! -f $OUT/scores/smoke_train.json ]; then
  log "smoke"
  rm -rf tmp/hv2/smoke && python method/pege_ddb.py variant --base checkpoints/pege_ddb --out tmp/hv2/smoke/pege_smoke --k 2 --depth 2 --grid 64 || exit 1
  python eval/train/evaluate.py --methods-root tmp/hv2/smoke --output $OUT/scores/smoke_train.json || exit 1
  grep -q '"ok"' $OUT/scores/smoke_train.json || { cat $OUT/scores/smoke_train.json; tail -50 logs/eval_train/pege_smoke/episode000.log; exit 1; }
fi

# ---- 4. official train / test scores (reference + baselines only)
if [ ! -f $OUT/details/train_reference.json ]; then
  log "evaluate_train"
  bash scripts/evaluate_train.sh --details $OUT/details/train_reference.json || exit 1
fi
if [ ! -f $OUT/details/test_reference.json ]; then
  log "evaluate_test"
  bash scripts/evaluate_test.sh --details $OUT/details/test_reference.json || exit 1
fi
cat scoring/scores_test.json

# ---- 5. hand-written one-step-lookahead rules (must not outrank the reference)
if [ ! -f $OUT/details/rules_test.json ]; then
  log "rules"
  R=tmp/hv2/rules; rm -rf $R
  python scripts/hardening/rule_policies.py build --out $R/rule_parent_label_w1 --kind parent_label --w 1.0 || exit 1
  python scripts/hardening/rule_policies.py build --out $R/rule_parent_features_tuned --kind parent_features --tune || exit 1
  python eval/train/evaluate.py --methods-root $R --output $OUT/scores/rules_train.json
  heval --methods-root $R --output $OUT/scores/rules_test.json --details $OUT/details/rules_test.json
  cat $OUT/scores/rules_test.json
fi

# ---- 6. shortcut probes (cheap, stdlib)
if [ ! -f $OUT/details/probes_test.json ]; then
  log "probes"
  P=tmp/hv2/probes; rm -rf $P; mkdir -p $P
  for m in random_control self_report invalid_action crash_midway hang hidden_file_read; do
    mkdir -p $P/$m
    echo "{\"command\": [\"python\", \"scripts/hardening/probe_policies.py\", \"$m\"]}" > $P/$m/policy.json
  done
  mkdir -p $P/forged_manifest
  echo '{"command": ["python", "scripts/hardening/probe_policies.py", "forged_manifest"], "gamma": 1.05, "budget": 310, "budget_fraction": 2.0, "experiment": "shortcut_unscored", "data": "/tmp/fake.json", "results": {"auc": 1e9}}' > $P/forged_manifest/policy.json
  mkdir -p $P/no_manifest
  echo '{"experiments":{"synthetic_main":{"results":{"random_control":{"auc":1e6,"reward_25pct":1e6,"reward_50pct":1e6,"reward_75pct":1e6}}}}}' > $OUT/scores/probes_test.json
  export SAPOG_STEP_TIMEOUT=20
  heval --methods-root $P --output $OUT/scores/probes_test.json --details $OUT/details/probes_test.json
  unset SAPOG_STEP_TIMEOUT
  # tampering with the lock fails every method in that run
  T=tmp/hv2/tamper; rm -rf $T
  for m in random_control unlock_read; do
    mkdir -p $T/$m
    echo "{\"command\": [\"python\", \"scripts/hardening/probe_policies.py\", \"$m\"]}" > $T/$m/policy.json
  done
  heval --methods-root $T --output $OUT/scores/tamper_test.json
  # interrupted evaluation must leave worst values, not stale scores
  echo '{"experiments":{"synthetic_main":{"results":{"random_control":{"auc":1e6}}}}}' > $OUT/scores/probes_interrupted_test.json
  chmod u+rwx eval/test
  timeout -s KILL 6 python eval/test/evaluate.py --methods-root $P --output $OUT/scores/probes_interrupted_test.json
  chmod u+rwx eval/test
  cat $OUT/scores/probes_interrupted_test.json $OUT/scores/tamper_test.json
fi

# ---- 7. FOG (privileged, diagnostic only) on validation
if [ ! -f $OUT/scores/fog_validation.json ]; then
  log "fog diagnostic"
  python method/fog_diagnostic.py --model-dir checkpoints/pege_ddb --out $OUT/scores/fog_validation.json
fi

# ---- 8. honest variants on the hidden test (each its own methods root)
run_variant() {  # name, then 'variant' or 'train', then method args
  local name=$1; shift
  local kind=$1; shift
  if [ -f $OUT/details/${name}_test.json ]; then return 0; fi
  log "variant $name"
  local dir=tmp/hv2/$name/$name
  if [ ! -f $dir/policy.json ]; then
    rm -rf tmp/hv2/$name
    if [ "$kind" = variant ]; then
      python method/pege_ddb.py variant --base checkpoints/pege_ddb --out $dir "$@" || return 1
    else
      python method/pege_ddb.py train --out $dir "$@" || return 1
    fi
  fi
  heval --methods-root tmp/hv2/$name --output $OUT/scores/${name}_test.json --details $OUT/details/${name}_test.json
  python scripts/hardening/summarize.py > $OUT/summary.txt 2>&1
}
python scripts/hardening/summarize.py > $OUT/summary.txt 2>&1
run_variant depth5 variant --depth 5
run_variant theta_parent_label variant --theta parent_label
run_variant no_expansion variant --depth 0
run_variant k2 variant --k 2
run_variant seed12 train --seed 12
run_variant seed13 train --seed 13
run_variant k48 variant --k 48
run_variant mean_plus_var variant --aggregate mean_plus_var
run_variant mode variant --aggregate mode
run_variant mean_minus_var variant --aggregate mean_minus_var

python scripts/hardening/summarize.py > $OUT/summary.txt 2>&1
cat $OUT/summary.txt
log "done"
