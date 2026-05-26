# Reproduction Milestones

**Current: method_runs**

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
