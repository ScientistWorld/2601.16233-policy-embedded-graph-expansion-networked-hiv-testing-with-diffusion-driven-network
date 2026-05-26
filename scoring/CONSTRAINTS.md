# Scientist Constraints

## Data

Use the provided `data/synthetic_hiv_forest.json` generator and five component-level folds. Do not add external labeled HIV network data, and do not split connected components across train and test. The original ICPSR dataset requires a user agreement, so this gym's fixed substitute is the comparable benchmark surface.

## Budget

Use the same full-population testing budget and `gamma = 0.99` when reporting `synthetic_main`. Reducing the tested population fraction can improve apparent early precision but changes the task.

For `synthetic_k_sweep` and `synthetic_aggregation`, keep the same folds, budget, discount factor, expansion depth, and oracle grid as the main run. Only the named ablation axis should change.

## Information

Policies may only act on discovered frontier nodes and information revealed by previous tests. They must not inspect labels of untested nodes or hidden future graph topology, except for the explicit fully observable upper-bound baseline.

## Compliance Checklist

- [ ] Uses the provided generated forest data and component-level folds
- [ ] Does not train on held-out fold labels
- [ ] Uses `gamma = 0.99` and the same budget definition
- [ ] Selects only currently discovered frontier nodes during evaluation
- [ ] Does not use hidden future graph topology for non-upper-bound methods
