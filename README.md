# Policy-Embedded Graph Expansion: Networked HIV Testing with Diffusion-Driven Network Samples

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2601.16233"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2601.16233-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2601.16233](https://arxiv.org/abs/2601.16233) |
|:---|:---|
| **Domain** | Sequential decision-making / public health network testing |
| **TL;DR** | The paper improves network-based HIV testing when the contact graph is revealed only through referrals by sampling hidden graph expansions inside the decision loop. |

## Summary

The task is to choose a testing order on partially observed contact forests so HIV-positive individuals are detected as early as possible. The paper proposes PEGE, a Monte Carlo decision framework that evaluates actions over sampled graph completions. Its DDB expansion model combines Gaussian Process branch counts with conditional diffusion-generated child covariates. On the paper's ICPSR HIV network benchmark, PEGE + DDB outperforms random, greedy, classifier, and DQN baselines in discounted reward.

## Problem

How should limited HIV tests be allocated sequentially when the contact network is only partially observed and grows through referrals? At each step, the decision maker can test one discovered but untested person, observes that person's binary HIV status, and discovers their adjacent referrals for future testing.

## The paper's method

The paper introduces Policy-Embedded Graph Expansion (PEGE), which samples possible hidden graph expansions and embeds those samples directly inside the sequential testing policy.

## Key results of the paper

- PEGE + DDB reports AUC 9.29 versus 8.45 for the strongest practical baseline.
- At 25%, 50%, and 75% testing budgets, PEGE + DDB reports discounted rewards of 8.02, 9.94, and 11.2.
- The fully observable Gittins policy remains an upper bound with AUC 12.0.

## Run it

```bash
pip install sciworld
sw research 2601.16233 --mode innovate --agent claude-code
sw evaluate 2601.16233
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

