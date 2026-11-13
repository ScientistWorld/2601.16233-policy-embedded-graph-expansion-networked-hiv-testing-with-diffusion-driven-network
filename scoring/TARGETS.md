# Evaluation Targets

## Primary: synthetic_main

Tests whether a sequential testing policy finds HIV-positive people early on held-out referral forests that are revealed only through its own tests.

- **Primary metric**: `auc`, the mean cumulative discounted reward over the full testing sequence (higher is better, coefficient 1.0)
- **Secondary metrics**: `reward_25pct` (0.7), `reward_50pct` (0.5), `reward_75pct` (0.4): cumulative discounted reward at those budget fractions (higher is better)
- **Constraints (eligibility, enforced by the evaluator)**: the fixed full budget `T = |V|`, `gamma = 0.99`, actions only on discovered untested people, and information limited to what earlier tests revealed. A violation, crash or stall scores the run as failed (0.0 in every metric), so no constraint can be traded for reward.
- **Reference to beat**: the reference method's row in `scoring/scores_test.json` / `scoring/reference.json`. The simple baselines (random, greedy neighbour, greedy classifier) are listed there as well.
- **How to run**: `bash scripts/evaluate_train.sh` (visible validation episodes) and `bash scripts/evaluate_test.sh` (held-out episodes; no arguments; scores every `checkpoints/<name>/policy.json`). `bash scripts/reproduce.sh` retrains the reference and baselines and runs both.
