# Output distillation for the frozen XSum prefix bridge

Classification: `output_distillation_joint_gate_failed`. This is not semantic summary accuracy.

| Adapter | Code agreement | All decoded IDs | Non-pad IDs | Content IDs |
|---|---:|---:|---:|---:|
| initial | 34.67% | 22.84% | 22.87% | 16.17% |
| mse | 35.18% | 22.59% | 22.64% | 16.68% |
| distill | 36.41% | 24.82% | 24.89% | 16.83% |

Content improvement over matched MSE continuation: 0.15% points.
Both arms started from the same 3.54M-parameter adapter and used 5,120 additional updates on the same 512 fit sources in the same order. All original models stayed frozen.
The distillation arm matched final code features for five epochs, then teacher code distributions for five epochs. The control continued input-representation MSE for ten epochs.
The 128 probe sources were excluded from adapter fitting but reused from the previous diagnostic; the released teacher may have seen them in its public training data. Inputs fit 512 positions and outputs remain 48 positions.
Exact reference agreement tests behavior preservation. Different summaries may be valid paraphrases, so agreement is not semantic accuracy. No application DEV/final cases, new semantic judgments or speed measurements were used. The separate main hybrid remains at 0/128 at its completed epoch1 and epoch2 DEV milestones.
