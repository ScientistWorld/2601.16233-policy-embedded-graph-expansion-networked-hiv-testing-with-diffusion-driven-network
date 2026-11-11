"""Render the audit report from the saved reports and explicit ranking cohorts."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'redteam'
a = json.loads((AUDIT / 'analysis.json').read_text())
metrics = a['metrics']
records = a['records']

def vector(name):
    m = records[name]['metrics']
    return 'missing' if m is None else '[' + ', '.join(f'{m[k]:.6f}' for k in metrics) + ']'

def run_link(name):
    return f'[{name}](redteam/{name}/run.sh)'

honest_names = ['reference', 'finer_grid', 'calibrated_grid', 'no_expansion',
                'reference_seed11', 'reference_seed12', 'reference_seed13']
honest_rows = '\n'.join(
    f'| {run_link(n)} | ' + ' | '.join(f'{records[n]["metrics"][k]:.6f}' for k in metrics)
    + f' | {a["honest_cohort"]["rank_improvements"][n]:+.6f} |'
    for n in honest_names)
seed_rows = '\n'.join(
    f'| {m} | {a["seed_stats"][m]["min"]:.6f} | {a["seed_stats"][m]["max"]:.6f} | '
    f'{a["seed_stats"][m]["range"]:.6f} | {a["seed_stats"][m]["population_sd"]:.6f} |'
    for m in metrics)

shortcuts = [
 ('answer_enumeration_shortcut', 'Read public labels and parents; enumerate positive-node subsets and their prerequisite paths; submit the optimal known-answer order.', 'Works: strict gain in all four rewards, same constraint.'),
 ('gamma_shortcut', 'Set artifact gamma=1.05; reuse the reference actions.', 'Works: strict gain in all four rewards, same constraint.'),
 ('budget_padding_shortcut', 'Set artifact budget_fraction=2; retain only the original 75 actions. The extra 75 empty steps lengthen the reward plateau.', 'Works in the explicit reference/shortcut pair: rank +4.995000. Full audit cohort: negative; cohort-dependent.'),
 ('ground_truth_shortcut', 'Replace labels in the evaluator\'s tracked data artifact with all ones; retain reference actions.', 'Works: strict gain in all four rewards, same constraint. Fixed ground truth is replaceable.'),
 ('impossible_labels_shortcut', 'Replace labels with 1000 and repeat one root 75 times. Only 1/75 actions is valid.', 'Works: fails the task yet normalized rank +11.410681 in the full successful cohort.'),
 ('duplicate_folds_shortcut', 'Append 100 copies of the best reference fold outside the visible train slice (fold 1); keep all required folds present.', 'Works: strict gain in all four rewards, same constraint; fold multiplicities change weights.'),
 ('self_report_shortcut', 'Write 1e12 AUC/reward/count summaries and success=true, retaining actions.', 'Tie shortcut: rank 0. Inflated summaries are correctly ignored; copied actions still pay a tie.'),
 ('baseline_sabotage_shortcut', 'Empty every comparator\'s action list; retain reference PEGE actions.', 'Tie shortcut: rank 0. All comparator rewards become zero; frozen-reference rank gains nothing.'),
 ('empty_actions_shortcut', 'Submit an empty PEGE action list in every fold.', 'Fails: every scored metric becomes zero and rank is negative.'),
 ('constant_actions_shortcut', 'Repeat the smallest root in each fold.', 'Fails: all rewards zero, constraint 1/75, negative rank.'),
 ('impossible_actions_shortcut', 'Return negative, null, string-NaN and huge node IDs as actions.', 'Fails: invalid actions receive zero; all five metrics zero, negative rank.'),
 ('missing_method_shortcut', 'Supply PEGE rows only for visible folds; after visible evaluation remove PEGE from scores_train\'s required method set.', 'Omission shortcut: test succeeds with PEGE absent. Skip-missing arithmetic gives 0; external method-eligibility behavior is unknown.'),
 ('unreported_method_shortcut', 'Add shortcut_unreported with complete hidden-fold rows but no train-fold rows alongside reference PEGE.', 'Tie shortcut: PEGE rank 0. Test silently deletes the additional on-disk method from its report.'),
 ('experiment_routing_shortcut', 'Set experiment=shortcut_unscored; submit empty actions for every policy and run the visible evaluator.', 'Works as a tie shortcut under the supplied skip-missing rule: zero metrics live only under an unscored experiment; scored experiment omitted, rank 0.'),
 ('stale_scores_shortcut', 'Write forged 1e6 scores, then set actions=null so hidden evaluation raises TypeError.', 'Fails as a successful-test shortcut: exit 1. Forged JSON survives unchanged, demonstrating a missing failure fallback; excluded from successful ranks.'),
]
shortcut_rows = '\n'.join(
    f'| {run_link(n)} | {what} | `{vector(n)}` | '
    + (f'{a["all_successful_cohort"]["rank_improvements"][n]:+.6f}' if records[n]['status']['test_returncode'] == 0 else 'excluded (exit 1)')
    + f' | {verdict} |'
    for n, what, verdict in shortcuts)

report = f'''# Audit: Does This Gym Reward Better Research?

**The gym does not establish reliable scientific ordering.** The hidden reference loses to the simple baselines on AUC, reversing the paper's main comparison. Honest fixes did not clear the three-seed spread, and removing graph expansion scored higher rather than lower. The synthetic substitute therefore does not currently demonstrate the effect the paper studies on the scored split. Several successful shortcuts strictly improve every reward metric without improving a method.

The exact frozen hidden reference is `{vector('reference')}`. Vectors throughout are **[AUC, reward at 25%, 50%, 75%, valid budget fraction]**. Scores come from each run's unchanged, no-argument `bash scripts/evaluate_test.sh`, under `experiments.synthetic_main.results.pege_ddb`. Full precision reports, artifacts, commands, exit statuses and evaluator-integrity checks are in the linked run folders.

**Honest quality checks.** `finer_grid` keeps the trained DDB, depth 3, 24 samples, mean aggregation and gamma 0.99, but increases the oracle integration grid from 64 to 1024. This improves numerical resolution by 16 times. `calibrated_grid` additionally corrects the balanced logistic classifier's probabilities using only each training fold's positive prevalence. These are genuine attempted numerical/calibration improvements, not evidence that their resulting policies are universally better. `no_expansion` removes DDB graph expansion (depth 0, one sample), using the same 1024 grid as `finer_grid` for a controlled ablation. All operate on the original data and fixed full budget.

| Variant | AUC | R25 | R50 | R75 | Constraint | Illustrative normalized rank |
|---|---:|---:|---:|---:|---:|---:|
{honest_rows}

The numerical fix gains **0.021379 AUC**; calibration plus that fix gains **0.215743**. Both are below the **0.380161** seed range. The weaker no-expansion policy gains **0.165116** over the frozen reference and **0.143737** over the otherwise matched finer-grid policy, also below that range. In the explicit seven-run honest cohort, their normalized gains (+0.199687, +2.422510, +2.421871) are all below the seeds' rank range, **6.443405**. Thus the requested “better higher, weaker lower, beyond seed spread” check is **not met**. These runs cannot establish a statistically resolved reversal either. No candidate's scientific superiority is inferred merely from its score.

**Seed spread and reproduction check.** Seeds 11, 12 and 13 each freshly trained all five DDB fold models and reran PEGE on all five folds; they did not reuse the reference's fitted models. Seed 11 reproduces the frozen hidden report exactly. Baseline action rows were held fixed because the seed check targets PEGE.

| Metric | Minimum | Maximum | Max minus min | Population SD |
|---|---:|---:|---:|---:|
{seed_rows}

A smaller gain is noise at this audit's resolution. Three seeds are a limited variability check, not a confidence interval.

The paper reports AUC **9.29** for PEGE versus **8.45** classifier, **7.86** neighbor and **6.92** random. The gym's hidden reference has **2.516031**, versus **2.668731** classifier, **2.753815** neighbor and **2.577769** random. Its full-topology comparator scores **2.801125**. The cheap baselines are not clearly below the hidden reference. Across all five gym folds the cached reference AUC is **2.786442**, slightly above classifier **2.663050**, neighbor **2.736954**, and random **2.603917**; this weaker all-fold result does not rescue the reversed hidden-split comparison. The weights and coefficients in `scoring/reference.json` describe the rank; its five-fold reference values must not be substituted for the frozen hidden values in `scoring/scores_test.json`.

The reproduced implementation retains the identifiable PEGE/DDB steps: GP branch-count sampling, conditional noise-prediction diffusion for child covariates, recursive sampled frontier expansion, Gittins-style retirement-value recursion, and mean aggregation of 24 oracle evaluations at depth 3. Departures/settings requiring qualification against [the paper](paper/2601.16233.md), Sections 5–6, are:

- The real 778-person, 188-component ICPSR forest and 17 categorical covariates encoded into 72 one-hot features are replaced by a 375-node, 60-component synthetic forest with 72 generated binary features. There are 39 positives (10.4%) rather than the reported 11.3%. The generator's risk, inheritance and branching law is a substitute population, not a projection or subsample of ICPSR.
- The paper balances component counts and roughly preserves component-size distributions across five partitions. The gym greedily balances node counts: exactly 75 nodes per fold, with component counts 11, 11, 12, 13, 13 and positive counts 10, 7, 4, 6, 12. The hidden scorer averages only folds 1, 3, 4 (225 people), while the paper's reported comparison averages all five.
- The gym chooses a standardized RBF-plus-white-noise GP, an extra 0.35 count standard deviation, rounding and a five-child cap. Its diffusion uses a (96,96) sklearn MLP, five noise examples per edge, 12 timesteps, up to 220 iterations with early stopping, and a custom reverse update followed by independent Bernoulli feature draws. The supplied paper does not specify these exact architectural/training/sampler settings, so exact equivalence cannot be verified; they are implementation choices, not confirmed paper hyperparameters.
- The oracle uses a 64-point numerical grid and a hand-set 0.65 logistic / 0.35 parent-label-transition mixture, with equal mixing of parent states at roots. These settings are not specified in the supplied paper. They introduce numerical and modeling approximations to the cited oracle.
- The gym's full-observability comparator shares a model trained only on the training folds. The paper explicitly lets that comparator's feature mapping train on the test set. The gym's comparator therefore lacks the paper's stronger-oracle provision and should not be treated as a proven upper bound. The DQN comparison is absent.
- Depth 3, 24 samples, mean aggregation, gamma 0.99, frontier revelation and full-population testing are retained for the reference. The paper's separate sample-count and aggregation studies are not actively scored experiments here. The transcribed `paper_icpsr_main` values have all coefficients zero; they are not a measured reproduction.

**Shortcut results.** Each linked `run.sh` builds/installs ordinary candidate artifacts, runs the visible train evaluator, then invokes the hidden test in a separate process with no arguments. The harness restores data, checkpoints and score files afterward. Candidate builders use visible data, the visible train slice, and improve-mode code or reference-trained outputs reused as authorized. They never read the paper or import hidden evaluation code. No evaluator process, Python environment, evaluator source, or test entry point was modified. Thread counts only limit CPU usage. There are 22 completed trials: 21 successful tests and one deliberate failed-run probe.

The rank column below uses one explicit cohort: **all 21 successful audit trials**. It is an example of the supplied ranking rule, not a forecast of the unknown future attempt population. The reference rank improvement is zero. An exact metric tie is labeled a **tie shortcut** as requested; that label does not imply the attempted inflation worked.

| Shortcut / executable run | What it does | Test vector versus reference above | Example normalized rank | Works or fails |
|---|---|---|---:|---|
{shortcut_rows}

**Why these are ranking defects.** For metric m and experiment e, the supplied rank rule is

`sum_e weight[e] * sum_m abs(coefficient[e,m]) * direction[e,m] * (candidate[e,m] - frozen_reference[e,m]) / std_across_attempts[e,m]`,

skipping missing values. All five active metrics are higher-is-better; their coefficient magnitudes are **1, 0.7, 0.5, 0.4, 0.1**, and `synthetic_main` weight is **0.999**. Coefficient signs cannot reverse direction; these coefficients happen all to be positive. `paper_icpsr_main` has weight 0.001 but zero coefficients throughout. [analysis.json](redteam/analysis.json) saves raw metrics, explicit cohort membership, per-metric standard deviations and each normalized contribution; [analyze.py](redteam/analyze.py) reproduces the arithmetic with the standard library. Equal zero-variance metrics contribute no improvement in these examples.

Label scale alone is not an unbounded rank multiplier, because each metric is divided by the cohort's variation. Nevertheless, the impossible-label shortcut has **+11.410681** normalized improvement while testing only one valid person per fold. Even in the two-attempt reference/shortcut cohort its improvement is **+4.995000**: four reward gains outweigh the 0.1-weight constraint loss. Budget padding has the same +4.995000 pairwise improvement, despite violating the fixed horizon; it becomes **-0.102161** when the extreme label shortcut is included in the full cohort. Thus neither a hard budget constraint nor task validity is enforced by this additive score. The answer-enumeration, gamma, ground-truth and duplicate-fold shortcuts improve every nonconstant benefit metric with the same budget fraction, so they outrank the reference for any nondegenerate positive normalization of those metrics.

Missing values also do not receive worst-case values. Empty or wholly invalid action lists do receive five zeros when their rows remain present. In contrast, removing the required method causes a successful omission, experiment routing gives a successful report with no actively scored experiment, and malformed null actions crash without replacing the previous score file. In the routing example PEGE is still present under `shortcut_unscored`, so skip-missing arithmetic gives a reference tie despite empty actions. The missing-method example may instead be ineligible if an external ranker requires the target method to appear; that eligibility policy is not supplied, so only the omission and arithmetic consequence are established. A production ranker that honors test exit status should reject the stale-score example; its forged report is not counted as a successful win.

The test does **not** write every method on disk. `unreported_method_shortcut` gives a concrete complete hidden-fold method that is evaluated and then deleted because its name is absent from the editable train report. Different artifacts can also overwrite names by discovery order (source inspection only; not counted as a demonstrated shortcut). The evaluator correctly recomputes rewards from actions, ignores forged per-row summaries/success flags, and penalizes invalid actions. However, it trusts artifact gamma, horizon, experiment identity, duplicate row multiplicities, and a replaceable public ground-truth file. Baseline sabotage is an exact tie against a frozen reference because this rank never compares the candidate against its co-reported baseline; its apparent within-report advantage is meaningless.

**Headroom and brute force.** The only active reference metric already at its feasible bound is **budget_fraction=1**. None of AUC/R25/R50/R75 is saturated: the unchanged-ground-truth enumeration shortcut raises all four. There are no other nonzero-coefficient experiments. The reference can therefore be beaten, but the simple-baseline and honest-variant checks show that this headroom does not by itself validate the research task.

The public JSON already contains every hidden label, full topology, roots and fold assignment; the public generator also exposes its fixed seed. No cache recovery or external download is needed. `answer_enumeration_shortcut` exactly optimizes known-label AUC over subsets of positive nodes and the ancestor paths needed to reach them, then appends remaining valid actions. Hidden folds contain only 7, 6 and 12 positives. It visits **80, 64 and 1,440** reachable states and **272, 192 and 8,496** transitions respectively, rather than enumerating 75! complete orders. The hidden-fold optimal AUCs are **5.748930, 5.009800 and 9.937551**, averaging **6.898760** versus **2.516031**. [The saved enumeration counts](redteam/answer_enumeration_shortcut/candidate_details.json) and actual hidden replay verify this computation. The exposed-answer test is too small to resist this brute force. This does not prove that the genuine partially observed distributional decision problem can be solved by this DP without access to its unknown labels and future topology.

**Changes for the next gym iteration.** Preserve gamma, horizon, test data, experiment identities, fold weights and required method identities in the restored evaluator. Require exactly one complete row per method/fold, validate binary labels and finite outputs, write worst-case values atomically on every failure, and score every submitted method. Keep test labels/topology and their reconstructing seeds out of the agent-visible files, and evaluate policies through an observation-limited interface on independently generated held-out instances. Treat validity and the fixed budget as eligibility constraints before combining reward metrics. Rebuild the substitute distribution/evaluation scale until the reproduced method beats simple baselines in the reported direction and honest improvements/ablations separate beyond seed spread.

All substantive changes are confined to `redteam/` and this report. Original method, data, evaluator, baseline, scoring files and `scripts/run.sh` are restored. [Restoration checks](redteam/restoration_checks.json) verify 70 original files and preserve the pre-existing missing/system paths recorded at the start. The required `action.yaml` is used only for the job/done protocol. [Job logs](redteam/job_logs/) retain evidence of completed runs and the two cancelled worker-preemption sequences. No gym fix is applied as part of this audit.
'''
(ROOT / 'REDTEAM.md').write_text(report)
