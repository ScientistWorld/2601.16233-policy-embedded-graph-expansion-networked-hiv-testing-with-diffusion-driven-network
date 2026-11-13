# Experiment Overview

## synthetic_main

Establishes how well a policy finds HIV-positive people early when testing a contact network that is revealed only through referrals. Each held-out episode is a forest of referral trees of about 155 people; only the seeds are visible at the start. Every test reveals one status and the tested person's referrals. Policies are compared at the same full budget (everyone is tested) and the same discount (`gamma = 0.99`), so the only thing that differs is the **order** of testing.

- The primary score (`auc`) rewards finding positives early across the whole testing sequence.
- The budget checkpoints (`reward_25pct`, `reward_50pct`, `reward_75pct`) show where in the sequence a policy gains or loses: early (scarce test kits) versus late.
- Validity is an eligibility condition, not a metric. A policy that acts on undiscovered or already-tested people, crashes or stalls is scored as failed (0 in every metric).

The simple reference points are random testing, testing referrals of positive people first, and ranking people by a covariate-only risk classifier. A policy that also reasons about the people who are not yet discovered should beat them. Testing with full knowledge of the hidden network would do far better still; on the visible episodes such an unattainable oracle scores about 50% above the reference.
