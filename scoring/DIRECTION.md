# Research Direction

## Research Problem

This benchmark studies sequential testing on partially observed contact forests. The central question is how to account for hidden future referrals while choosing the next discovered person to test.

## Core Contribution

The paper's contribution is distribution-aware graph expansion inside the decision loop for incrementally revealed graphs. Improvements should still address uncertainty about hidden graph growth and node attributes in a sequential frontier policy.

## Approach Scope

In-scope work may change the graph expansion model, the action scoring oracle, aggregation across sampled futures, or training procedures for these components. The method should remain a network-aware sequential policy that operates under partial observability.

## Out of Bounds

Do not replace the problem with static node classification, full-graph active search, or a policy that assumes future topology or hidden labels are available. Do not beat the benchmark by expanding the dataset, changing the test budget, or using the fully observable upper-bound information in a practical method.
