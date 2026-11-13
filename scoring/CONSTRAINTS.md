# Scientist Constraints

## Data

Learn only from `data/train_forest.json` (the public training forest; its five component-level folds are there for internal validation). Tune on the visible validation episodes in `data/validation_episodes.json` if you like. Do not add external labelled HIV network data. The held-out episodes come from the same distribution but are generated inside the hidden evaluator. Do not try to recover them: no reading or unlocking `eval/test/`, and no inspecting the evaluator process.

## Budget

Every person in an episode is tested (`T = |V|`) and `gamma = 0.99`. Both are fixed by the evaluator, and anything a method writes about them is ignored.

## Information

At each step a policy may use only the roots, the people revealed by its own earlier tests (their covariates and referrer), and the labels of the people it tested in **this** episode. Each evaluation episode is an independent instance. Do not persist revealed labels or other episode information (files, caches, background processes) from one evaluation episode to another. Learning happens from the training data before evaluation, not from the held-out episodes.

## Compliance Checklist

- [ ] Models trained only on `data/train_forest.json` (optionally tuned on the visible validation episodes)
- [ ] The policy acts only on discovered, untested people and finishes every episode
- [ ] No use of undiscovered people, unrevealed labels or the hidden evaluator's files or memory
- [ ] No state carried between evaluation episodes
- [ ] Same budget (`T = |V|`) and `gamma = 0.99` as the evaluator
