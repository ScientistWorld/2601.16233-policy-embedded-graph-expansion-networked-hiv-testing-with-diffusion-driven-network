# Progress

## What Works

- Paper analysis is complete enough to identify the core method: PEGE with DDB expansion, evaluated on sequential HIV testing over forest contact graphs.
- Implemented a portable synthetic forest substitute because the ICPSR data requires a user agreement.
- Implemented DDB with Gaussian Process branch counts and conditional denoising diffusion for child covariates.
- Implemented PEGE action selection with sampled graph expansions and a forest oracle.
- Implemented Random, Greedy Neighbor, Greedy Classifier, and Fully Observable Gittins comparison policies in the same run artifact.
- Greedy Classifier is now the paper-aligned balanced linear SVM comparator, while DDB keeps a separate calibrated logistic model for label probabilities inside graph expansion/oracle scoring.
- Implemented an evaluation script that reads result artifacts and writes `scoring/scores.json` without importing from `method/`.
- A CPU smoke job completed end-to-end on 3 folds with `k = 8`; `python validate.py --compare` passed.
- A corrected 5-fold `k = 24` run completed cleanly and wrote validated scores.

## Results

Corrected 3-fold smoke result on the synthetic substitute (`k = 8`, Gittins grid 64):

| Method | AUC | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| PEGE + DDB | 2.793 | 1.540 | 3.032 | 3.899 |
| Greedy Classifier | 3.104 | 2.119 | 3.353 | 4.418 |
| Greedy Neighbor | 2.541 | 0.902 | 2.753 | 3.612 |
| Random | 2.574 | 1.202 | 2.709 | 4.158 |
| Fully Observable Gittins | 5.762 | 5.929 | 6.469 | 6.469 |

PEGE + DDB beats Random and Greedy Neighbor on primary AUC in this smoke setting but trails Greedy Classifier, so this is still `method_runs`. The next run includes two fidelity fixes: frontier Gittins scores are conditioned on observed parent labels, and Greedy Classifier uses the balanced linear SVM comparator specified by the paper.

Corrected 5-fold result on the synthetic substitute (`k = 24`, `d = 3`, Gittins grid 64):

| Method | AUC | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| PEGE + DDB | 2.743 | 1.470 | 2.805 | 3.929 |
| Greedy Classifier | 2.663 | 1.619 | 2.512 | 3.910 |
| Greedy Neighbor | 2.737 | 0.920 | 2.945 | 3.854 |
| Random | 2.604 | 1.095 | 2.601 | 4.087 |
| Fully Observable Gittins | 6.447 | 6.733 | 7.221 | 7.221 |

PEGE + DDB beats all practical baselines on the primary AUC metric under the fixed full-budget constraint, which supports the core claim that sampled graph expansion inside the decision loop improves sequential testing. The result is not `core_claim_plus`: PEGE trails some baselines at individual budget checkpoints and beats Greedy Neighbor by only `0.006` AUC.

## Remaining

- Higher milestones would require additional settings beyond the fixed synthetic substitute, such as alternative budgets or multiple synthetic graph regimes, while preserving the same partial-observability protocol.

## Issues

- The original ICPSR 22140 HIV network is public only after completing an ICPSR user agreement, so it cannot be fetched automatically into this gym.
- The login environment does not currently have `numpy`; dependencies are declared in `environment/setup.sh` and will be installed by the container setup step.
- First CPU smoke job failed before method execution because top-level packages were not on `PYTHONPATH` when executing `method/pege_ddb.py` as a script. Fixed by adding package markers and exporting `PYTHONPATH=/home/user`.
- Second CPU smoke job trained the first DDB fold, then failed because NumPy 2.4 removed `np.trapz`. Fixed with a local trapezoid calculation.
- The 5-fold run completed artifacts but exited `-9`, likely due to the one-hour CPU test limit. The produced artifacts validate locally, so the next job only evaluates existing outputs.
- A validation-only retry failed before executing code because `environment/container.sif` had been removed while clearing quota. Local validation still passes on the completed artifacts, and no method or scoring changes are needed.
- Validator post-check rejected direct comparison between synthetic-scale scores and paper Table 1 values. Fixed by making `scoring/reference.json` describe the synthetic benchmark scale and moving the original paper table to `notes/paper_table1.md`.
- A follow-up code review found that the first implementation's oracle was not close enough to the cited Gittins-index computation. The implementation now uses a numerical binary branching-bandit Gittins recursion from the cited equations, and the prior synthetic result is treated as superseded rather than a `core_claim` result.
- The first corrected run was interrupted by a Hive disk-quota heartbeat failure after training several fold models. The method now writes fold/policy results incrementally and resumes from existing DDB fold checkpoints, so infrastructure interruption no longer discards all progress.
- The corrected 3-fold smoke run completed cleanly, but PEGE+DDB did not beat the classifier baseline. That exposed two implementation fidelity gaps, now fixed before the next job.
- The final corrected run validates cleanly, but the reproduced primary-metric margin is narrow on the synthetic substitute.

## Deviations from Paper

- **Dataset**: The paper uses ICPSR 22140 with 778 nodes, 188 connected components, and 72 binary covariates. This reproduction uses a procedural synthetic HIV-like forest with the same partial-observability protocol because ICPSR access requires a user agreement.
- **Oracle details**: The paper uses the Gittins index policy from Choo et al. 2025. This implementation uses the cited binary branching-bandit retirement-value recursion on a discretized grid, with learned DDB conditionals substituting for unavailable ICPSR transition estimates.
- **Scale for first corrected job**: The completed smoke job used 3 folds and `k = 8`. The next job uses the paper's 5 folds and `k = 24` on the synthetic substitute.
