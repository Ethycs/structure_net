# Frozen-Llama bridge to the released XSum prefix

Classification: `prefix_bridge_preservation_failed`. This is a mechanism-preservation experiment, not semantic summary accuracy.

| Prefix | Code agreement | All decoded IDs | Non-pad IDs | Native content IDs | Projected MSE |
|---|---:|---:|---:|---:|---:|
| bridge | 34.67% | 22.84% | 22.87% | 16.17% | 0.052160715 |
| mean | 20.91% | 0.78% | 0.72% | 0.72% | 0.1035188 |

Bridge MSE / mean-control MSE: 0.503877. Empty native outputs: 0.
A 3,540,352-parameter MLP alone was trained for 5,120 updates on 512 public TRAIN sources. The 128 probe sources were excluded from bridge fitting; the released XSum teacher may have seen them during pretraining.
Every source fits the native 512-position prefix without truncation. The native output frame remains 48 positions. The Llama body, NCM generator and codec stayed fixed. No application DEV/final case, new semantic judgment, or speed measurement was used.
Passing this gate would motivate further range and task-quality testing. It cannot establish 95% complete-summary accuracy. The separate current hybrid still has 0/128 at both completed application DEV milestones.
