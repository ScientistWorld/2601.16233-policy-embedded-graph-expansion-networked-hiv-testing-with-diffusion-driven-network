# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

### [2026-05-26 08:02] - method_runs
- Superseded the previous `core_claim` result after review found two fidelity issues: generated child covariates were ignored for synthetic-node label probabilities, and the FOG upper bound used unrevealed labels.
- Patched the method so generated covariates influence oracle probabilities and FOG observes full topology/features without future-label leakage.
- Prepared a clean five-fold rerun; `core_claim` will only be restored if the corrected artifact supports it.

### [2026-05-26 09:01] - method_runs
- Job `22b35548-c3e` made partial corrected-run progress but exited `-9` after a Hive preemption/orphan event.
- Corrected artifacts now contain complete folds 0-2 for all scored policies.
- Removed the checkpoint wipe from `scripts/run.sh` so the next submission resumes folds 3-4 and then runs evaluation/validation.

### [2026-05-26 09:18] - core_claim
- Job `4ce1f310-1e9` resumed the corrected benchmark, completed folds 3-4, and wrote full five-fold artifacts for all scored policies.
- `python validate.py --compare` passed in the job.
- Corrected PEGE+DDB achieved the best practical partial-observation AUC on `synthetic_main` (2.786), ahead of Greedy Neighbor (2.737), Greedy Classifier (2.663), and Random (2.604).

### [2026-05-26 09:25] - core_claim
- Final verification kept the milestone at `core_claim`.
- Removed the pending ablation submission from the scored gym surface because no completed ablation artifacts were present.
- Confirmed the completed five-fold `synthetic_main` result remains the validated benchmark.

### [2026-11-14] - core_claim
- Hardening iteration. The reference was restored to the paper's DDB/oracle specification after a diffusion-schedule bug was found. The interactive hidden test was enlarged to 120 episodes and locked while policies run.
- PEGE+DDB beats every practical baseline on the hidden test (AUC 7.195 vs 5.975 for the greedy classifier; 18× the seed range). The paper's k and aggregation trends are reproduced.
