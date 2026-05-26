# Overview

- **Paper ID:** 2601.16233
- **Title:** Policy-Embedded Graph Expansion: Networked HIV Testing with Diffusion-Driven Network Samples
- **Domain:** Sequential decision-making / public health network testing
- **TL;DR:** The paper improves network-based HIV testing when the contact graph is revealed only through referrals by sampling hidden graph expansions inside the decision loop.

## Short Summary

The task is to choose a testing order on partially observed contact forests so HIV-positive individuals are detected as early as possible. The paper proposes PEGE, a Monte Carlo decision framework that evaluates actions over sampled graph completions. Its DDB expansion model combines Gaussian Process branch counts with conditional diffusion-generated child covariates. On the paper's ICPSR HIV network benchmark, PEGE + DDB outperforms random, greedy, classifier, and DQN baselines in discounted reward.

## Key Results

- PEGE + DDB reports AUC 9.29 versus 8.45 for the strongest practical baseline.
- At 25%, 50%, and 75% testing budgets, PEGE + DDB reports discounted rewards of 8.02, 9.94, and 11.2.
- The fully observable Gittins policy remains an upper bound with AUC 12.0.

---

# Problem Definition

## Research Question

How should limited HIV tests be allocated sequentially when the contact network is only partially observed and grows through referrals? At each step, the decision maker can test one discovered but untested person, observes that person's binary HIV status, and discovers their adjacent referrals for future testing.

## Why It Matters

Testing resources and staff time are limited. A good sequential testing policy finds HIV-positive individuals earlier, so treatment and prevention can start sooner.

## Success Criteria

A successful policy earns high discounted cumulative positive detections on held-out forest contact graphs. It should work from initially visible root nodes, use only information revealed so far, and preserve the fixed testing budget used in evaluation.

---

# Paper's Method

## Key Contribution

The paper introduces Policy-Embedded Graph Expansion (PEGE), which samples possible hidden graph expansions and embeds those samples directly inside the sequential testing policy.

## Approach

For each current frontier node, PEGE samples `k` possible future graph expansions, evaluates the frontier action under an oracle on each sampled graph, averages the oracle scores, and tests the node with the highest mean score.

The expansion model is Dynamics-Driven Branching (DDB). DDB trains on parent-child dynamics in forest graphs: a Gaussian Process model samples branch counts from parent covariates, and a conditional denoising diffusion model samples child covariates from parent covariates. Recursive sampling from frontier nodes creates stochastic forest-preserving graph completions up to depth `d`.

## Main Claims

The primary claim is that PEGE + DDB improves discounted HIV detections over practical baselines on incrementally revealed contact forests. Secondary claims are that the method is sample efficient in the number of expansions and that mean aggregation of oracle scores beats heuristic aggregation variants.

---

# Evaluation

## Metrics

The primary metric is AUC under the cumulative discounted reward curve, where each positive test contributes `gamma^t` at testing step `t` and `gamma = 0.99`. Higher AUC means positive individuals are found earlier across the testing sequence.

Secondary metrics are cumulative discounted reward at 25%, 50%, and 75% of the testing budget. The constraint metric is the tested budget fraction; policies are compared at the same full-population budget.

## Baselines and Targets

The portable benchmark compares submitted policies against simple reference policies and a full-topology comparator using the same output artifact format. `scoring/reference.json` includes the original ICPSR Table 1 values for context under `paper_icpsr_main` and the scored portable synthetic benchmark under `synthetic_main`.

## Evaluation Protocol

The benchmark uses five component-level folds. Components are never split across train and test. For each fold, a static held-out graph is tested from its root nodes, results are averaged with weights proportional to held-out node count, and `scripts/evaluate.sh` reads produced result artifacts to write `scoring/scores.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 5 total (1 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 8 total (3 completed, 4 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-26 07:35] - core_claim
- The corrected 5-fold `k=24`, `d=3` synthetic benchmark completed and passed `python validate.py --compare`.
- PEGE+DDB achieved primary AUC `2.743`, beating Random (`2.604`), Greedy Classifier (`2.663`), and Greedy Neighbor (`2.737`) under the same full-budget constraint. The margin over Greedy Neighbor is narrow, so this restores `core_claim` but not a higher milestone.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 9 total (4 completed, 4 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 9 total (4 completed, 4 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 9 total (4 completed, 4 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 4 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 4 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-26 07:23] - method_runs
- Implemented the paper's PEGE loop with sampled DDB graph expansions, a GPR branch-count model, a conditional denoising diffusion child-covariate model, and a Gittins-style forest oracle.
- Added a portable synthetic forest substitute for the ICPSR HIV network, which requires a data-use agreement.
- Prior jobs showed the method can execute end-to-end, but the current checkpoint artifact is incomplete and must be rerun before claiming `core_claim`.

### [2026-05-26 07:43] - core_claim
- Job `4fd15c8c-403` completed the clean five-fold synthetic benchmark with `k = 24`, expansion depth `3`, and Gittins grid `64`.
- The regenerated `scoring/scores.json` is based on complete fold artifacts for PEGE+DDB, Greedy Classifier, Greedy Neighbor, Random, and the fully observable Gittins upper bound.
- PEGE+DDB achieves the highest primary AUC among practical partial-observation policies on the portable synthetic benchmark, supporting the paper's central claim under the documented ICPSR data-access substitution.
- `python validate.py --compare` passed in the job, including import-separation and evaluator-output checks.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 12 total (6 completed, 5 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**
