# Hardening report (iteration 11)

This iteration ran in two sessions. Session 1 (commit `4e05bb5`, job `58b18399`) replaced the audited gym with an interactive hidden test on a new synthetic world. Session 2 (this report's final state, jobs `a9e434d7`, `9a594716`, `374bde8f`) closed the runtime leak that remained, fixed the reproduced method where it departed from the paper, enlarged the test, and re-audited. Numbers are hidden-test values `[AUC, R25, R50, R75]`. Paired differences are against the reference on the same episodes, given as mean ± standard error. Raw per-episode data: `notes/hardening/details/` (final), `notes/hardening/round1/` (session 1), `notes/hardening/summary.{txt,json}`.

## 1. Do better results now score higher?

**Before (REDTEAM, old gym).** No. The reference (2.516 AUC) lost to all three simple baselines (classifier 2.669, neighbour 2.754, random 2.578). Removing graph expansion scored *higher* (+0.165). Honest fixes (+0.021, +0.216) were below the 0.380 seed range.

**Session 1 (interactive test, 80 episodes, previous DDB).** The ordering was right in the paper's direction (reference 6.750 > classifier 6.269), and weaker variants scored lower. But a three-line one-step-lookahead rule beat the reference by **+0.78 ± 0.21** AUC (offline replay on the same 80 episodes). The Gittins oracle fed the rule's simple ingredients also beat it (+0.66 ± 0.17). The reference was broken, not the task.

**Final (120 episodes, fixed DDB).** Better variants score higher and weaker ones lower, but the honest gains are small:

| variant | AUC | R25 | R50 | R75 | ΔAUC vs ref | illustrative rank |
|---|---:|---:|---:|---:|---:|---:|
| k = 48 expansions (paper's sample-count study) | 7.291 | 5.040 | 8.604 | 9.882 | **+0.096 ± 0.061** | +0.195 |
| **reference PEGE+DDB (seed 11)** | **7.195** | **5.001** | **8.356** | **9.807** | 0 | 0 |
| seed 13 | 7.181 | 5.033 | 8.442 | 9.803 | −0.013 ± 0.064 | +0.033 |
| oracle planning discount 0.98 / 0.97 | 7.155 / 7.156 | 4.936 / 4.889 | 8.299 / 8.322 | 9.789 / 9.781 | −0.040 ± 0.031 / −0.039 ± 0.037 | −0.09 / −0.12 |
| seed 12 | 7.128 | 4.870 | 8.392 | 9.741 | −0.067 ± 0.064 | −0.142 |
| depth 5 | 7.074 | 4.895 | 8.151 | 9.677 | −0.121 ± 0.068 | −0.259 |
| theta without referrer covariates (previous θ) | 7.064 | 4.860 | 8.226 | 9.646 | −0.131 ± 0.130 | −0.274 |
| θ refit on DDB-generated children | 6.984 | 4.949 | 7.982 | 9.470 | −0.211 ± 0.088 | −0.403 |
| k = 2 | 6.769 | 4.550 | 7.801 | 9.272 | −0.426 ± 0.107 | −0.929 |
| mean + variance aggregation | 6.369 | 4.172 | 7.310 | 9.000 | −0.826 ± 0.166 | −1.703 |
| greedy classifier (baseline) | 5.975 | 4.077 | 6.626 | 8.303 | −1.220 ± 0.198 | −2.467 |
| no expansion (d = 0) | 5.805 | 3.799 | 6.505 | 8.201 | −1.390 ± 0.202 | −2.839 |
| greedy neighbour (baseline) | 3.418 | 1.563 | 3.329 | 5.262 | −3.777 ± 0.258 | −7.868 |
| random (baseline) | 3.141 | 1.436 | 2.927 | 4.870 | −4.054 ± 0.266 | −8.394 |

The illustrative rank is Σ|coef|·Δ/std over the non-probe rows of this table and the two rules. The seed range over seeds 11/12/13 is **0.067** AUC.

- **The paper's main claim** is reproduced in the paper's direction: PEGE+DDB beats the greedy classifier by 1.22 ± 0.20 AUC (+20%; the paper reports +10%). It is 18× the seed range.
- **The paper's secondary claims** also hold. Performance drops with few expansions (k = 2: −0.43) and plateaus above 24 (k = 48: +0.10). Mean aggregation beats mean + variance (−0.83; the paper reports 65% of mean).
- **Restoring θ to the cited oracle's form** (referrer covariates) is worth +0.13 over the previous θ.
- **The best honest variant, k = 48,** gains +0.096 ± 0.061. That exceeds the seed range but is only 1.6 standard errors. Headroom is demonstrated only weakly (§5).

## 2. What was fixed

### Evaluator and test
- *Session 1:* The test is interactive. A method is a policy (`checkpoints/<name>/policy.json`) played over JSON lines on held-out episodes. Those episodes are generated in memory inside `eval/test/` from hidden seeds and never touch disk. The evaluator fixes gamma, the budget (T = |V|, everyone tested), the episodes, the metrics and the aggregation; nothing the method writes is read as a score. Any invalid action, crash or timeout fails the run with 0 in every metric. A worst-value placeholder is written first and scores are written atomically. Every method directory on disk is scored.
- *Session 2:* While policies run, `eval/test/` is set to mode 0 and the scorer marks itself non-dumpable. If the directory's ctime changes during the run (anyone chmods it), every method in that run fails. `evaluate_test.sh` restores the permissions first, in case a killed run left the directory locked.
- *Session 2:* The test grows from 80 to 120 episodes. The first 80 are unchanged, so before/after comparisons are paired.

### Reproduced method (restoring the paper's version; §6 of the paper, Choo et al. 2025)
The DDB diffusion was broken. Its 12-step schedule ended at ᾱ_T ≈ 0.61, but sampling started from N(0, I), so generated children were mostly noise. Diagnostics (`notes/hardening/scores/ddb_diagnostics.json`, validation episodes):

| | previous DDB | fixed DDB |
|---|---:|---:|
| generated children that are valid one-hot people | 0% (23.4 ones vs 17) | 100% |
| mean abs. gap of bit marginals vs real children | 0.111 | 0.034 |
| θ on generated children of positive referrers (real: ≈0.50) | 0.20 | 0.31 |
| GPR count correlation with the true count | 0.27 (children) | 0.33 (\|N(v)\|) |

The fixes:
- **Diffusion:** a 40-step cosine schedule, ancestral DDPM sampling with x₀ clipping, and decoding to one answer per categorical question.
- **GPR:** it now models |N(v)| as Section 5 states. A non-root person's successors are |N(v)| − 1.
- **θ:** it now uses the referrer's covariates as well as the referrer's label. Choo et al.'s pairwise potentials use both endpoints' covariates.

PEGE settings stay at the paper's k = 24, d = 3, mean aggregation and γ = 0.99. On the 80 shared episodes the fixed reference gains **+0.706 ± 0.199** AUC over the session-1 reference.

## 3. Shortcuts: fix and score before → after

| shortcut (REDTEAM name / probe) | fix | before | after |
|---|---|---:|---:|
| answer enumeration over public test labels | test labels exist only in memory inside `eval/test/`; actions reveal labels one at a time | 6.899 (vs ref 2.516) | not expressible |
| reading the hidden generator at test time (`hidden_file_read`) | `eval/test/` unreadable while policies run | 13.204 (vs ref 6.750) | 3.141 (= random) |
| unlocking it first (`unlock_read`) | ctime tamper check fails every method in that run | — | 0 (all methods failed) |
| gamma = 1.05, budget padding, experiment routing, forged fields in `policy.json` (`forged_manifest`) | constants fixed by the evaluator; manifest only supplies a command | gamma 20.98, padding 3.98, routing tie | 3.141 (= random control) |
| replaced or impossible ground-truth labels | labels generated in memory, never read from disk | 29.76 / 986.7 | not expressible |
| duplicated folds | episodes and weights fixed by the evaluator | 3.814 | not expressible |
| self-reported scores (`self_report`) | replies other than the action are ignored | tie | 3.141 (= random control) |
| empty, constant or invalid actions (`invalid_action`) | any invalid action fails the run | 0 / 0 / 0 | 0 (failed) |
| crash / hang (`crash_midway`, `hang`) | fail the run; 15-min per-step timeout | — | 0 (failed) |
| missing / unreported method (`no_manifest`) | every directory is scored; missing `policy.json` gives worst values | omitted (rank 0) | 0 (failed) |
| stale scores after a crash (interrupted run) | placeholder of worst values first, atomic write | forged 1e6 kept | 0 in every metric |
| hand-written one-step-lookahead rule | reproduction fixed to the paper's method (§2) | +0.78 ± 0.21 | +0.118 ± 0.129 (fixed weight), −0.019 ± 0.123 (weight tuned on validation) |

## 4. Shortcuts left, and why

- **Hand-written lookahead rule (a statistical tie, not closed).** p + γ·ĉ·(p·q1 + (1−p)·q0), with a logistic p, a ridge referral count ĉ, and training-forest child positivity q1/q0. With a fixed weight it scores 7.313 against 7.195 (+0.118 ± 0.129, rank +0.36). The version tuned on the visible validation episodes, which is what an agent would pick, scores −0.019 ± 0.123. On the visible slice both rules score above the reference (6.61 / 6.71 vs 6.36, 20 episodes). In this synthetic world the paper's method, faithfully reproduced, matches a calibrated one-step lookahead but does not beat it clearly. Closing the gap would require redesigning the world so that multi-step distributional lookahead matters more. I did not do that within this iteration's budget.
- **Cross-episode memory.** A policy could save revealed test labels to a file or `/tmp` and reuse them in later episodes, which amounts to online learning on the test set. Stopping that needs an OS sandbox the container does not provide. `scoring/CONSTRAINTS.md` and `briefing/evaluation.md` forbid it, and a grader can detect it from any file writes in the policy's serve path.
- **Root inside the test container.** A policy running as root bypasses the permission lock (the ctime check still catches a chmod). The hidden test runs as the user.
- **Compute scaling.** More expansions (k = 48) is a small legitimate gain, bounded by the 15-min per-step timeout. It follows the paper's own sample-count trend and is not a shortcut.

## 5. Reference score before → after, and headroom

- **Reference:** 2.516 (REDTEAM, old data, not comparable) → 6.750 (session 1, 80 episodes) → **7.195** (final, 120 episodes). On the shared 80 episodes: +0.706 ± 0.199.
- **Bounds:** the only bounded metric (`budget_fraction`, stuck at 1) was removed; the budget is now an eligibility rule. No scored metric is at its bound. Rewards are far below the unattainable full-observability comparator: on the visible episodes the privileged FOG diagnostic scores 9.56 AUC against the reference's 6.36.
- **Better variants found:** k = 48 (+0.096 ± 0.061, above the 0.067 seed range) and the fixed-weight rule (+0.118 ± 0.129). Both are small. I tried discount matching, deeper expansion, a generated-child calibration of θ and (offline) sibling-aware risk; none helped. I added no new experiment. The headroom that honest variants demonstrate is weak, and that is the main open limitation of this gym.
- **Brute force:** none applies. Labels and the hidden topology are never available before a test, and the generator is unreadable at test time.

## 6. Weights

`synthetic_main` weight 1.0; coefficients unchanged (AUC 1.0, R25 0.7, R50 0.5, R75 0.4). They follow the paper's headline metrics: total AUC, and detections with 25% of the population tested. The paper's Table 1 is kept unchanged under `_paper_icpsr_main`, since it is on a different dataset.

## 7. What is not fixed or not verified

- The rule tie and the weak headroom (§4, §5).
- The ICPSR network is replaced by a synthetic world; scores are not on the paper's scale.
- θ is a directed logistic approximation of Choo et al.'s pairwise MRF, which they fit by pseudo-likelihood; the MRF itself is not implemented.
- The DQN baseline is absent.
- The paper's mode and mean − variance aggregations were rerun only in session 1, with the previous DDB: mode +0.027 ± 0.122, mean − variance −1.917 ± 0.331.
- The final job reproduces the reference end to end (`scripts/run.sh`) and checks it against the recorded scores (`scripts/hardening/compare_reference.py`). Its outcome is recorded below.
