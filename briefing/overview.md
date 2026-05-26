# Overview

- **Paper ID:** 2601.16233
- **Title:** Policy-Embedded Graph Expansion: Networked HIV Testing with Diffusion-Driven Network Samples
- **Domain:** Sequential decision-making / public health network testing
- **TL;DR:** The paper improves network-based HIV testing when the contact graph is revealed only through referrals by sampling hidden graph expansions inside the decision loop.

## Short Summary

The task is to choose a testing order on partially observed contact forests so HIV-positive individuals are detected as early as possible. The paper proposes PEGE, a Monte Carlo decision framework that evaluates actions over sampled graph completions. Its DDB expansion model combines Gaussian Process branch counts with conditional diffusion-generated child covariates. On the paper's ICPSR HIV network benchmark, PEGE + DDB outperforms random, greedy, classifier, and DQN baselines in discounted reward.

## Key Results

- PEGE + DDB reports AUC 9.29 versus 8.45 for the strongest practical baseline.
- At 25%, 50%, and 75% testing budgets, PEGE + DDB reports discounted rewards of 8.02, 9.94, and 11.2.
- The fully observable Gittins policy remains an upper bound with AUC 12.0.
