# Progress

## What Works

- Paper analysis is complete enough to identify the core method: PEGE with DDB expansion, evaluated on sequential HIV testing over forest contact graphs.
- Implemented a portable synthetic forest substitute because the ICPSR data requires a user agreement.
- Implemented DDB with Gaussian Process branch counts and conditional denoising diffusion for child covariates.
- Implemented PEGE action selection with sampled graph expansions and a forest oracle.
- Implemented Random, Greedy Neighbor, Greedy Classifier, and Fully Observable Gittins comparison policies in the same run artifact.
- Implemented an evaluation script that reads result artifacts and writes `scoring/scores.json` without importing from `method/`.
- A CPU smoke job completed end-to-end on 3 folds with `k = 8`; `python validate.py --compare` passed.
- A 5-fold `k = 24` run produced complete PEGE + DDB artifacts and `scores.json` before the scheduler killed the shell process.

## Results

3-fold smoke result on the synthetic substitute:

| Method | AUC | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| PEGE + DDB | 3.169 | 2.441 | 3.476 | 3.916 |
| Greedy Classifier | 3.104 | 2.119 | 3.353 | 4.418 |
| Greedy Neighbor | 2.541 | 0.902 | 2.753 | 3.612 |
| Random | 2.574 | 1.202 | 2.709 | 4.158 |
| Fully Observable Gittins | 5.634 | 5.871 | 6.411 | 6.411 |

The method beats the strongest practical baseline by AUC in the smoke setting, but the margin is small and the 75% budget value trails Greedy Classifier. A 5-fold `k = 24` run is queued to strengthen the result before claiming `core_claim`.

5-fold `k = 24` result on the synthetic substitute:

| Method | AUC | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| PEGE + DDB | 2.839 | 1.485 | 3.002 | 4.002 |
| Greedy Classifier | 2.718 | 1.452 | 2.774 | 4.156 |
| Greedy Neighbor | 2.737 | 0.920 | 2.945 | 3.854 |
| Random | 2.604 | 1.095 | 2.601 | 4.087 |
| Fully Observable Gittins | 6.302 | 6.668 | 7.155 | 7.155 |

PEGE + DDB beats all practical baselines on the primary AUC metric and at 25% and 50% budgets. It trails Random and Greedy Classifier at 75%, so this is a core-claim reproduction rather than a broad or near-complete reproduction.

## Remaining

- Run a short validation-only job so the scheduler records a clean successful completion for the 5-fold artifacts.

## Issues

- The original ICPSR 22140 HIV network is public only after completing an ICPSR user agreement, so it cannot be fetched automatically into this gym.
- The login environment does not currently have `numpy`; dependencies are declared in `environment/setup.sh` and will be installed by the container setup step.
- First CPU smoke job failed before method execution because top-level packages were not on `PYTHONPATH` when executing `method/pege_ddb.py` as a script. Fixed by adding package markers and exporting `PYTHONPATH=/home/user`.
- Second CPU smoke job trained the first DDB fold, then failed because NumPy 2.4 removed `np.trapz`. Fixed with a local trapezoid calculation.
- The 5-fold run completed artifacts but exited `-9`, likely due to the one-hour CPU test limit. The produced artifacts validate locally, so the next job only evaluates existing outputs.
- A validation-only retry failed before executing code because `environment/container.sif` had been removed while clearing quota. Local validation still passes on the completed artifacts, and no method or scoring changes are needed.

## Deviations from Paper

- **Dataset**: The paper uses ICPSR 22140 with 778 nodes, 188 connected components, and 72 binary covariates. This reproduction uses a procedural synthetic HIV-like forest with the same partial-observability protocol because ICPSR access requires a user agreement.
- **Oracle details**: The paper uses the Gittins index policy from Choo et al. 2025. This implementation uses a forest dynamic-priority oracle that scores immediate predicted positivity plus discounted descendant opportunity, preserving the oracle role inside PEGE but not claiming exact parity with the unpublished referenced implementation.
- **Scale for first job**: The first job uses 3 folds and `k = 8` for a smoke test. The paper's main setting is 5 folds and `k = 24`.
