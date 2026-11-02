# Measure learning on the public TRAIN pairs

Exploratory diagnostic ncm-public-pair-fit-probe-v1. The existing sixteen-case
TRAIN probe measures reviewed application anchors. At public epoch2 those
anchors have been presented about79 times each on average, while each public
pair has been presented exactly twice. Its98.36% rollout code agreement does
not establish fit on the public population. Application semantic DEV remains
0/128. The original public run and matched task-supervision runs stay unchanged.

Before looking at model outputs, select32 of all162,353 admitted public TRAIN
pairs by SHA256 of the fixed string public-pair-fit-271142:<case_id>, breaking
ties by case ID. Retain the original complete source and target, native frame,
oracle codes and original FP32 features. No DEV, component or final source enters
this diagnostic. Public XSum references are not individually factual gold.

Compare the exact completed BROAD and public epoch2 checkpoints, using the
existing reference diagnostic and source-only generation path. For each model,
evaluate correct sources and a fixed one-place cyclic source permutation,
holding each target fixed. Report teacher-forced and rollout code agreement,
per-level agreement and native nonpadding word agreement. Report changes within
this fixed public sample; comparisons with the earlier application controls are
descriptive and confounded by population and exposure differences. Neither
exact code agreement nor reference-token agreement is semantic task accuracy.
Alternative valid summaries can disagree with the reference. No fitted-code
threshold qualifies the95% goal or alone proves that undertraining is causal.

Use physicalGPU2 RTX2060SUPER, original FP32 weights/features and inference,
TF32 off and deterministic algorithms. Leave both existing GPU0/GPU1 units and
unrelated GPU2 processes untouched. Require at least3.5GB CUDA free memory before
loading, and peak own allocation no more than3.5GB. Stream one feature matrix
per requested case; do not materialize the feature corpus. No optimization,
gradient, model update, teacher/judge request or inference-speed claim is made.

First qualify CPU contracts. Then run a real two-case pilot on the first
selected case and the selected population's longest source, reproducing every
output exactly under both checkpoints. Require both existing training jobs to
have begun real updates before the pilot; preserve service identities across
the diagnostic. Full32 runs require the successful actual pilot receipt and
must reproduce its matched outputs. Failures stop the dependent stage and remain
append-only. No memory or equality tolerance fallback is allowed.

After full execution, independently replay raw teacher/rollout/word counts,
source permutations, checkpoint identities and fixed selection on CPU. Preserve
all raw outputs and artifacts, with actual terminal receipts between stages.
Report the descriptive learning result and the unchanged semantic failure
separately. Roots are20261027_public_pair_fit_{pilot,run,audit}_v1 beneath the
existing ncm_paper_regime artifact directory. This plan changes no ongoing
training schedule, checkpoint selection, judging lineage or sealed-final gate.
