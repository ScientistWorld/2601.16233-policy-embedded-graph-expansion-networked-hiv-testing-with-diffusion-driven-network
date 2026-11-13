# Research Direction

## Research Problem

This benchmark studies sequential testing on partially observed contact forests. The network grows only as tested people refer contacts. The central question is how to value a discovered person when much of the reason to test them lies in the people they will reveal, who are not yet visible.

## Core Contribution

The central research direction is uncertainty-aware sequential testing on incrementally revealed networks. Improvements should come from better reasoning about the unobserved part of the network and the labels in it, and from better decisions under that uncertainty, using only the small training forest.

## Approach Scope

In-scope work may change how the hidden part of the network is modelled, how uncertainty about it is represented, how frontier actions are valued, how exploration and exploitation are balanced, and how model components are trained from the training forest.

## Out of Bounds

Do not replace the problem with static node classification or full-graph active search. Do not assume future topology or hidden labels are available. Do not use held-out episode information beyond what the policy's own tests reveal in that episode, and carry nothing between evaluation episodes. Do not change the test budget, the discount or the data.
