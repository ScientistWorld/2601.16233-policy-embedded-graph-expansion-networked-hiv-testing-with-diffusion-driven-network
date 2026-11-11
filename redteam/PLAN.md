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

Job 1 (5b1446ae-a7f) and job 2 (a04752f4-ce6) were cancelled after
worker-drain preemptions left their logs idle for approximately 40–50 minutes.
All completed per-trial reports and per-fold model artifacts were retained.
Job 2 finished the no-expansion ablation and all three fresh seed reruns.
The final job resumes only calibrated_grid (one fold already saved), on a
one-hour CPU test node to avoid losing progress to another worker drain.
