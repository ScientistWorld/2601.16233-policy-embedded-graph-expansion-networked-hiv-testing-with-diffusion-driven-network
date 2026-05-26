# Paper Table 1 Reference

The original paper reports these values on the ICPSR HIV network. That dataset
requires a user agreement, so `scoring/reference.json` uses the portable
synthetic benchmark scale instead.

| Algorithm | AUC | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| Random | 6.92 | 5.10 | 8.11 | 9.13 |
| Greedy Neighbor | 7.86 | 6.70 | 8.75 | 9.91 |
| Greedy Classifier | 8.45 | 7.12 | 9.32 | 10.10 |
| DQN | 7.78 | 6.11 | 8.81 | 9.75 |
| PEGE + DDB | 9.29 | 8.02 | 9.94 | 11.20 |
| Fully Observable Gittins | 12.00 | 11.10 | 13.30 | 13.70 |
