# Paper's Method

## Key Contribution

The paper introduces Policy-Embedded Graph Expansion (PEGE), which samples possible hidden graph expansions and embeds those samples directly inside the sequential testing policy.

## Approach

For each current frontier node, PEGE samples `k` possible future graph expansions, evaluates the frontier action under an oracle on each sampled graph, averages the oracle scores, and tests the node with the highest mean score.

The expansion model is Dynamics-Driven Branching (DDB). DDB trains on parent-child dynamics in forest graphs: a Gaussian Process model samples branch counts from parent covariates, and a conditional denoising diffusion model samples child covariates from parent covariates. Recursive sampling from frontier nodes creates stochastic forest-preserving graph completions up to depth `d`.

## Main Claims

The primary claim is that PEGE + DDB improves discounted HIV detections over practical baselines on incrementally revealed contact forests. Secondary claims are that the method is sample efficient in the number of expansions and that mean aggregation of oracle scores beats heuristic aggregation variants.
