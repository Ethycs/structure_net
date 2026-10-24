# Packed-training versus sequential-generation code-path diagnostic

2026-10-24. Hypothesis `ncm-native655-history-path-parity-v1`.
Systems/mechanism diagnostic on original TRAIN cases, never semantic accuracy.
The source-only native655 checkpoint already failed0/128 DEV. Before another
adaptation, check whether its packed teacher-forced forward agrees with the
native sequential generation path when supplied the SAME earlier-level codes.
The source hook skips a one-token prefix; code inspection confirms that the
eager native generator prepends that prefix at every level, including the
root. This potential skip-root bug is not established. Measure actual logits.

Use the pinned completed native655 checkpoint SHA49a661bf..., exact original
TRAIN first/longest-source cases, full FP32 features and original clean641
codes. Parent body/MLPs remain frozen and absent. Freeze all model parameters;
no optimizer, gradient update, external judge, DEV or final data. PhysicalGPU1
RTX2060/logical0, FP32, TF32off; require2GBfree and enforce2GB peak allocated.
This small two-case cell is itself a CUDA systems pilot, not a full quality run.

Capture packed teacher_logits. Temporarily replace only the native sampler
with an oracle-code supplier, retaining each level's actual inference logits.
Restore the original sampler even on failure; check all641 supplied codes
returned unchanged, every native level present, model hashes unchanged and
source contexts cleared. Compare packed and sequential logits with rtol/atol
1e-4 and report max absolute differences and argmax disagreements per level.
Numerical mismatches are diagnostic, not an automatic explanation for failure.

Also record source-only free codes, clean-target CE and code agreement under
clean versus generated histories, and source-swapped teacher logits. The root
must not depend on later history under the native structural mask; report its
clean/generated difference. These are two TRAIN examples, not generalization
or task accuracy. Save full raw logits/codes for independent CPU metric replay.
Fresh root `20261024_native655_history_path_parity_v1`; preserve actual tool
exit and all prior failures. No live producer source is changed.
