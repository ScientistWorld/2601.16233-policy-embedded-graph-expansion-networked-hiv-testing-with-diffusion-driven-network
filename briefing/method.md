# Paper's Method

## Key Contribution

The paper introduces Policy-Embedded Graph Expansion (PEGE), which samples possible hidden graph expansions and embeds those samples directly inside the sequential testing policy.

## Approach

For each current frontier node, PEGE samples `k` possible future graph expansions, evaluates the frontier action under an oracle on each sampled graph, averages the oracle scores, and tests the node with the highest mean score.

The expansion model is Dynamics-Driven Branching (DDB). DDB trains on parent-child dynamics in forest graphs: a Gaussian Process model samples branch counts from parent covariates, and a conditional denoising diffusion model samples child covariates from parent covariates. Recursive sampling from frontier nodes creates stochastic forest-preserving graph completions up to depth `d`.

## Main Claims

The primary claim is that PEGE + DDB improves discounted HIV detections over practical baselines on incrementally revealed contact forests. Secondary claims are that the method is sample efficient in the number of expansions and that mean aggregation of oracle scores beats heuristic aggregation variants.

## Reproduction in this gym

- **DDB.** A Gaussian Process regressor (RBF + white-noise kernel on standardized covariates) gives a Gaussian over a person's number of neighbours `|N(v)|`, as in Section 5. A non-root person's successors are `|N(v)| - 1`, because one neighbour is their referrer. A conditional denoising diffusion model (epsilon-prediction MLP 96x96, 40-step cosine noise schedule, ancestral sampling) generates a child's 72 covariates from the parent's. Each sample is decoded to exactly one answer per categorical question.
- **Oracle.** The Gittins index policy for forests (Choo et al. 2025): a retirement-value recursion with Whittle's product formula over a node's children, each child conditioned on its parent's label. The feature mapping theta is a logistic model of P(HIV+ | own covariates, referrer's covariates, referrer's label). Choo et al.'s pairwise potentials use the covariates of both endpoints of an edge.
- **PEGE.** `k = 24` expansions of depth `d = 3` per step, mean aggregation of the oracle's indices, `gamma = 0.99`, budget `T = |V|`, training seed 11.
- **Data.** The ICPSR network needs a signed user agreement, so the gym uses a synthetic referral-forest substitute at the paper's per-fold scale. It has latent risk and activity traits inherited along referrals and 17 categorical answers (72 one-hot bits). The training forest has 778 people.
