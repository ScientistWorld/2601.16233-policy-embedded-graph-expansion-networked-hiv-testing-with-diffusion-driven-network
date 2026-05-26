# Experiment Overview

## synthetic_main

This experiment measures whether a policy can prioritize testing on incrementally revealed forest contact graphs. It compares cumulative positive detections under a fixed full-population budget and reports how much reward is accumulated early in the testing sequence.

## synthetic_k_sweep

This experiment measures how the policy changes as more sampled futures are used before each testing decision. It reports each sampled-future setting relative to the `k=12` setting so the result tracks the paper's sample-efficiency claim.

## synthetic_aggregation

This experiment compares the score aggregation rules evaluated by the paper while holding the testing protocol fixed. It reports each rule relative to mean aggregation, which is the paper's selected rule.
