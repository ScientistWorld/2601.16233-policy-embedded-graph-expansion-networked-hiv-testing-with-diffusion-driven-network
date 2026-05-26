# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped.
