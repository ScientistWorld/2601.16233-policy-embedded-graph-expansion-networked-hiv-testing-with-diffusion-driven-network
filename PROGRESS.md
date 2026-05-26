# Progress

## What Works

- Paper analysis is complete enough to identify the core method: PEGE with DDB expansion, evaluated on sequential HIV testing over forest contact graphs.
- Implemented a portable synthetic forest substitute because the ICPSR data requires a user agreement.
- Implemented DDB with Gaussian Process branch counts and conditional denoising diffusion for child covariates.
- Implemented PEGE action selection with sampled graph expansions and a forest oracle.
- Implemented Random, Greedy Neighbor, Greedy Classifier, and Fully Observable Gittins comparison policies in the same run artifact.
- Implemented an evaluation script that reads result artifacts and writes `scoring/scores.json` without importing from `method/`.

## Results

No compute job has completed yet. The first submitted job will run a 3-fold smoke reproduction and validate the produced scores.

## Remaining

- Run the method end-to-end in the compute container.
- Inspect whether PEGE + DDB beats the practical baselines on the synthetic substitute.
- If the core claim holds, expand from the smoke setting to all 5 folds and `k = 24`.

## Issues

- The original ICPSR 22140 HIV network is public only after completing an ICPSR user agreement, so it cannot be fetched automatically into this gym.
- The login environment does not currently have `numpy`; dependencies are declared in `environment/setup.sh` and will be installed by the container setup step.
- First CPU smoke job failed before method execution because top-level packages were not on `PYTHONPATH` when executing `method/pege_ddb.py` as a script. Fixed by adding package markers and exporting `PYTHONPATH=/home/user`.

## Deviations from Paper

- **Dataset**: The paper uses ICPSR 22140 with 778 nodes, 188 connected components, and 72 binary covariates. This reproduction uses a procedural synthetic HIV-like forest with the same partial-observability protocol because ICPSR access requires a user agreement.
- **Oracle details**: The paper uses the Gittins index policy from Choo et al. 2025. This implementation uses a forest dynamic-priority oracle that scores immediate predicted positivity plus discounted descendant opportunity, preserving the oracle role inside PEGE but not claiming exact parity with the unpublished referenced implementation.
- **Scale for first job**: The first job uses 3 folds and `k = 8` for a smoke test. The paper's main setting is 5 folds and `k = 24`.
