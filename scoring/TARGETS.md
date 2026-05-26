# Evaluation Targets

## Primary: synthetic_main

Tests whether a sequential testing policy finds positive individuals early on held-out forest contact graphs.

- **Metric**: `auc` (higher is better)
- **Secondary metrics**: `reward_25pct`, `reward_50pct`, `reward_75pct` (higher is better)
- **Constraint metric**: `budget_fraction` (lower is better only as a guard; comparisons should use the same full budget)
- **How to run**: `bash scripts/reproduce.sh`, or evaluate existing artifacts with `bash scripts/evaluate.sh`
