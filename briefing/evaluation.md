# Evaluation

## Metrics

The primary metric is AUC under the cumulative discounted reward curve, where each positive test contributes `gamma^t` at testing step `t` and `gamma = 0.99`. Higher AUC means positive individuals are found earlier across the testing sequence.

Secondary metrics are cumulative discounted reward at 25%, 50%, and 75% of the testing budget. The constraint metric is the tested budget fraction; policies are compared at the same full-population budget.

## Baselines and Targets

The paper reports Random, Greedy Neighbor, Greedy Classifier, DQN, the proposed policy, and a fully observable Gittins upper bound. Because this gym uses a synthetic substitute with a different reward scale, `scoring/reference.json` records the portable benchmark reference values and `notes/paper_table1.md` preserves the paper's original Table 1.

## Evaluation Protocol

The benchmark uses five component-level folds. Components are never split across train and test. For each fold, a static held-out graph is tested from its root nodes, results are averaged with weights proportional to held-out node count, and `scripts/evaluate.sh` reads produced result artifacts to write `scoring/scores.json`.
