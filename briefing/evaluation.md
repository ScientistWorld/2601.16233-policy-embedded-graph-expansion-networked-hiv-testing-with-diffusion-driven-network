# Evaluation

## What is measured

A submitted method is a **sequential testing policy**. It is played interactively on referral forests (episodes) of about 155 people with about 11% HIV prevalence. Each episode starts with the root people (recruited seeds) visible. At every step the policy names one discovered, untested person. The evaluator then reveals that person's HIV status and the people they referred (their covariates and who referred them). The policy must keep testing until every person in the episode has been tested (budget `T = |V|`).

Each positive test at step `t` (0-based) earns `gamma^t`, with `gamma = 0.99`. From the cumulative discounted reward curve the evaluator computes:

| Metric | Meaning | Coefficient |
|---|---|---:|
| `auc` (primary) | mean of the cumulative discounted reward curve over the `T` steps (area under the curve / `T`) | 1.0 |
| `reward_25pct` | cumulative discounted reward after 25% of the people are tested | 0.7 |
| `reward_50pct` | ... after 50% | 0.5 |
| `reward_75pct` | ... after 75% | 0.4 |

All four are higher-is-better. Episode values are averaged with weights proportional to the number of people per episode. Finding positives earlier raises every metric.

## Slices

- **Train / visible slice**: the 20 episodes in `data/validation_episodes.json`, scored by `bash scripts/evaluate_train.sh` into `scoring/scores_train.json` (and by `bash scripts/evaluate.sh` into `scoring/scores.json`).
- **Held-out test slice**: 120 episodes drawn from the same distribution and of the same size. They are generated inside the hidden evaluator and never written to disk. `bash scripts/evaluate_test.sh` (no arguments) scores every method under `checkpoints/` into `scoring/scores_test.json`.

Both slices use the same protocol, metrics and aggregation. Learning material is `data/train_forest.json`: 778 labelled people in referral trees, with five component-level folds for internal validation.

## Interface

Every directory `checkpoints/<name>/` containing a `policy.json` is a method:

```json
{"command": ["python", "path/to/policy.py", "--model-dir", "checkpoints/<name>"]}
```

The evaluator runs the command from the workspace root once per episode and exchanges one JSON object per line over stdin/stdout. The protocol is documented, and implemented for policies, in `data/sapog_protocol.py`: `serve(choose)` handles the bookkeeping. Reference policies for the simple baselines are in `baseline/policies.py`.

## Rules the evaluator enforces

- Gamma, the budget, the episodes, the metrics and the aggregation are fixed by the evaluator. Nothing a method writes (extra fields in `policy.json` or in its replies, files on disk) is read as a score.
- A run **fails**, getting the worst value (0.0) in every metric, if `policy.json` is missing or invalid, the policy crashes or times out (15 min per step), or it names someone who is not a discovered, untested person.
- A policy only ever sees roots, the people revealed by its own tests, and the labels of the people it tested. The held-out generator is unreadable while policies run. Tampering with it fails every method in that run.
- Each episode is an independent instance: a policy must not carry information (for example revealed labels) from one evaluation episode to another.

## Reference points

`scoring/reference.json` holds the scored results of the reference method and the paper's simple baselines (random, greedy neighbour, greedy classifier) on the held-out slice. It also keeps the paper's own ICPSR table under `_paper_icpsr_main`, on a different dataset, for context only.
