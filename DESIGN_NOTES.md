# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. They dispatch to separate slice evaluators and write `scoring/scores_train.json` / `scoring/scores_test.json`.
- Checked for hardcoded method defaults. The slice scripts discover `checkpoints/**/results.json` when artifacts are not explicitly passed.
- Checked for train/test cross-imports. The slice evaluators duplicate their replay logic and do not import each other.
- Checked score stubs. `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Reviewed `scoring/reference.json` for paper-agreement metrics. None of the active metrics are paper-deviation or fit-to-published metrics.

## Changes Made

- Fixed the main shortcut risk: both slice evaluators now replay submitted `actions` against the tracked SAPOG data and recompute the reward curve, AUC, budget checkpoints, and valid budget fraction. They no longer trust artifact-reported `auc` or `cumulative_discounted_reward` fields.
- The replay enforces the incremental frontier process: invalid, duplicate, out-of-fold, or not-yet-discovered actions consume a step with zero reward and do not reveal children.
- Changed the held-out slice from folds `{1, 3}` to `{1, 3, 4}` so the unused fifth fold contributes to final test discrimination.
- Neutralized the non-redistributable ICPSR table as reference-only scoring metadata by setting its experiment weight and metric coefficients to `0.0`.
- Set the portable `synthetic_main` weight to `1.0`.
- Added required metric descriptions in `scoring/reference.json`.
- Changed the active `budget_fraction` direction to higher-is-better with a small positive coefficient. It now measures the fraction of the allowed budget consumed by valid sequential actions, which penalizes early-stopping or invalid-action shortcuts instead of rewarding them.

## Runtime Concerns for Runner Iteration

- The train/test evaluators now import `numpy` and `data.sapog_data`; these are already used elsewhere in the project, but the runner should confirm the final container still has the same environment.
- Because the benchmark data file contains labels for all five synthetic folds, an agent that can see all data could still construct oracle action artifacts for every fold. The hidden `eval/test/` split helps with metric/code overfit, but a stronger future design would use a separate hidden test dataset or labels unavailable during train-side development.
- The test slice has three folds while the visible train slice has two. This improves held-out stability, but train/test variance should be inspected when scores are generated.
- No experiments were run in this design pass, so the runner should verify that recomputed train/test numbers differ from the prior self-reported-artifact scores only through intended replay and split changes.

## Runner Results

- Job `54556eec-4d7` ran `scripts/evaluate_train.sh /home/user/checkpoints` and `scripts/evaluate_test.sh /home/user/checkpoints` on CPU and completed successfully.
- Generated `scoring/scores_train.json` and `scoring/scores_test.json` contain the same five methods for `synthetic_main`: `pege_ddb`, `greedy_classifier`, `greedy_neighbor`, `random`, and `fully_observable_gittins`.
- The proposed method's replayed AUC is comparable across the split (`2.6943` train, `2.7755` test), while the fully observable oracle remains a clear upper bound (`5.7699` train, `6.8978` test).
- Benefit metrics vary across methods and train/test values are not byte-identical. `budget_fraction` is `1.0` for every current method because all existing artifacts contain valid full-budget action sequences; it remains a low-weight validity guard against future invalid-action or early-stopping shortcuts.
