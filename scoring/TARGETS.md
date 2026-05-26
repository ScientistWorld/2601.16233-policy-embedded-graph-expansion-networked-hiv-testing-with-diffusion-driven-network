# Evaluation Targets

## Primary: synthetic_main

Tests whether a sequential testing policy finds positive individuals early on held-out forest contact graphs.

- **Metric**: `auc` (higher is better)
- **Secondary metrics**: `reward_25pct`, `reward_50pct`, `reward_75pct` (higher is better)
- **Constraint metric**: `budget_fraction` (lower is better only as a guard; comparisons should use the same full budget)
- **How to run**: `bash scripts/reproduce.sh`, or evaluate existing artifacts with `bash scripts/evaluate.sh`

## Secondary: synthetic_k_sweep

Tests whether the PEGE decision loop responds to the number of sampled graph expansions in the same way as the paper's sample-count ablation.

- **Metric**: `auc_lift_vs_k12_pct` (higher is better)
- **Constraint metric**: `budget_fraction`
- **How to run**: `bash scripts/ablation.sh`

## Secondary: synthetic_aggregation

Tests whether mean aggregation of oracle scores remains the strongest PEGE aggregation rule among the paper's tested heuristics.

- **Metric**: `auc_ratio_to_mean` (higher is better)
- **Constraint metric**: `budget_fraction`
- **How to run**: `bash scripts/ablation.sh`
