# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-26 06:20] - method_runs
- Implemented PEGE with DDB: Gaussian Process branch-count model, conditional denoising diffusion for child covariates, recursive forest expansion from frontier nodes, and Monte Carlo policy embedding.
- Implemented SAPOG-style sequential testing on a portable synthetic HIV-like forest substitute.

### [2026-05-26 06:55] - method_runs
- A 5-fold synthetic benchmark completed with the first PEGE+DDB implementation, but code review found the oracle was a recursive priority proxy rather than the paper's cited Gittins index recursion.
- Replaced the oracle with a numerical binary branching-bandit Gittins recursion from the cited Choo et al. equations and queued a new run. The previous result is not used to claim `core_claim`.

### [2026-05-26 07:00] - method_runs
- Corrected run was interrupted by a Hive disk-quota heartbeat failure before writing final results.
- Added resumable fold/policy artifact writing and a smaller corrected smoke configuration (`k=8`, 3 folds, Gittins grid 64) to get a valid result before scaling back toward the paper's `k=24`, 5-fold setting.

### [2026-05-26 07:20] - method_runs
- The corrected 3-fold smoke job completed and produced valid scores, but PEGE+DDB still trailed the classifier baseline on primary AUC (`2.793` vs. `3.104`), so it is not a `core_claim` result.
- Fixed two fidelity issues before the next run: PEGE now conditions frontier Gittins scores on observed parent labels, and Greedy Classifier now uses the paper's balanced linear SVM comparator rather than the DDB logistic label model.

## Stop Justification
- Not stopped. A scheduler job is being submitted for the corrected 5-fold `k=24` PEGE+DDB run.
