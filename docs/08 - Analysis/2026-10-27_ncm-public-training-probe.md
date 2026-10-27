# Public epoch2: fixed TRAIN-control diagnostic

Both semantic DEV milestones remain 0/128. This sixteen-case TRAIN diagnostic is not task accuracy.

| Model | Source | Teacher code agreement | Rollout code agreement | Native-word agreement |
|---|---|---:|---:|---:|
| BROAD | matched | 89.65% | 40.65% | 19.15% |
| BROAD | shuffled | 75.37% | 10.41% | 7.50% |
| epoch2 | matched | 99.64% | 98.36% | 98.23% |
| epoch2 | shuffled | 88.76% | 10.01% | 6.09% |

Classification: `TRAIN_rollout_improved`. Matched rollout correct-code change: 5919.
Native-word agreement includes non-padding special tokens. These controls were used in training; there is no held-out generalization claim.
All weights stayed fixed during this diagnostic. The epoch4 curriculum continued unchanged on the RTX2060. No semantic judge was called and no speed was measured.
