# Train/Test Evaluation Design Notes

Current design (iteration 11; see `HARDENING.md` for the audit and numbers).

- **Methods are policies.** Each `checkpoints/<name>/policy.json` names a command. Both evaluators run it once per episode from the workspace root and play the episode over JSON lines. The protocol is in `data/sapog_protocol.py`. Every method directory on disk is scored, and no method name is hardcoded.
- **Train slice:** `eval/train/evaluate.py` plays the 20 visible episodes in `data/validation_episodes.json` and writes `scoring/scores_train.json` (or `scores.json` via `scripts/evaluate.sh`).
- **Test slice:** `eval/test/evaluate.py` plays 120 held-out episodes, generated in memory by `eval/test/world.py` from hidden seeds. While policies run, `eval/test/` is unreadable and the scorer is non-dumpable. A ctime change on the directory fails every method in that run.
- **Fixed by the evaluators:** gamma 0.99, budget T = |V| (everyone tested), the episodes, the metrics (AUC, R25, R50, R75) and the people-weighted aggregation. Any invalid action, crash or timeout fails the run with 0 in every metric. A worst-value placeholder is written first, then results are written atomically.
- `eval/train` and `eval/test` share no code, and neither imports `method/`.
- `data/train_forest.json` (778 people) and `data/validation_episodes.json` were written once by `eval/test/world.py --write-public` from public seeds. They are committed, so the hidden generator is not needed to use them.
