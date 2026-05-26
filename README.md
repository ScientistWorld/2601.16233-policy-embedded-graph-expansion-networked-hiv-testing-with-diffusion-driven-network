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

The paper reports Random, Greedy Neighbor, Greedy Classifier, DQN, the proposed policy, and a fully observable Gittins upper bound. Because this gym uses a synthetic substitute with a different reward scale, `scoring/reference.json` records the portable benchmark reference values and `notes/paper_table1.md` preserves the paper's original Table 1.

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


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
