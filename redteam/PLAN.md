Audit the unmodified hidden scorer using no-argument `bash scripts/evaluate_test.sh`
in a separate process for every trial. No evaluator/process/environment edits are
part of any candidate. Candidate builders use the visible data, visible train
evaluator, and (for honest variants) improve-mode reference implementation only.

Run order: original reference; artifact-only shortcuts and failure checks;
honest numerical-resolution and calibration variants; no-expansion ablation;
three complete reference reruns with seeds 11, 12, and 13. Save each result
immediately, restore checkpoint discovery/data/scoring after each run, and keep
all audit outputs under redteam. Training and inference run only in a CPU job.

Original artifacts are reused only where a trial changes scoring inputs rather
than the policy. Reference seed reruns retrain all five fold models and rerun
all five folds; they do not resume the original fitted models.
