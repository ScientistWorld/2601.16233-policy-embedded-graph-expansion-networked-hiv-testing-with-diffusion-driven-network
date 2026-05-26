# Research Direction

## Research Problem

This benchmark studies sequential testing on partially observed contact forests. The central question is how to account for hidden future referrals while choosing the next discovered person to test.

## Core Contribution

The central research direction is uncertainty-aware sequential testing on incrementally revealed networks. Improvements should still address hidden future referrals and unknown node attributes while choosing among currently available frontier nodes.

## Approach Scope

In-scope work may change how uncertainty is represented, how frontier actions are valued, how exploration and exploitation are balanced, or how model components are trained. The method should remain a network-aware sequential policy that operates under partial observability.

## Out of Bounds

Do not replace the problem with static node classification, full-graph active search, or a policy that assumes future topology or hidden labels are available. Do not beat the benchmark by expanding the dataset, changing the test budget, or using the fully observable upper-bound information in a practical method.
