# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-26 05:28] - none
- Started reproduction work from an empty scaffold.
- Read the paper and identified PEGE + DDB as the core algorithm.
- Implemented the first portable reproduction environment, but no compute job has completed yet.

### [2026-05-26 05:43] - none
- First CPU smoke job installed dependencies and generated data, then failed on `ModuleNotFoundError: data`.
- Fixed package import path handling and synthetic prevalence calibration.

### [2026-05-26 05:49] - none
- Second CPU smoke job generated 375 nodes with 10.4% positive prevalence and trained the first DDB fold.
- Job failed at metric computation due to a removed NumPy API; replaced `np.trapz` with a manual AUC calculation.

### [2026-05-26 05:56] - method_runs
- CPU smoke job completed PEGE + DDB training, rollout, artifact evaluation, and validation on 3 folds with `k = 8`.
- PEGE + DDB beat the strongest practical baseline by AUC on the smoke setting, but the margin is not yet strong enough to stop at `core_claim`.

### [2026-05-26 06:19] - core_claim
- 5-fold `k = 24` PEGE + DDB run produced complete artifacts and `scores.json`.
- PEGE + DDB beat all practical baselines on primary AUC and at 25% and 50% budgets on the synthetic forest substitute.
- Fully Observable Gittins remains a stronger upper bound, as expected.

### [2026-05-26 06:27] - core_claim
- Validation-only scheduler retry failed before running code because the generated container image had been deleted to recover quota.
- Re-ran artifact evaluation and `python validate.py --compare` locally; all checks pass with the completed 5-fold scores.

## Stop Justification
- Not stopped.
