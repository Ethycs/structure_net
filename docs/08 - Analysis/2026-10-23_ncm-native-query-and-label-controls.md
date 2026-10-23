# Native-query lifecycle works; student quality remains unmeasured

Date: 2026-10-23. Exploratory, `systems_passed_teacher_joint_capacity_gate_failed`.
The95%student summary goal remains unmet. These controls are neither a paper
training replication nor a speed benchmark.

The shared-query3B control retains all28native blocks and freezes every original
weight. Its initial true-previous-word numerical comparison reached99.21875%top1
agreement and relativeL2error0.01966365. That denominator includes256query
positions, including padding; it is **not summary accuracy or exact identity**.
The original backward OOM remains failed, with its completed update count unknown.
The explicit v2 memory recovery passed two updates, immutable sourceKV, frozen
weights and exact new-weight reload at8,908,633,088bytes peak allocation.

The actual five-concept-level NCM soft-input pilot passed two real updates and
frozen weights/books, conditional-residual, source-cache and free-output reload
checks. It adds8,637,696trainable parameters to the original model. Its text is
unusable after two updates; passing lifecycle checks establishes no competence.
Peak allocation was8,787,542,528bytes. No additional1Bdecoder or parent verifier
was used. The full soft-input-only training configuration was deferred.

The revised native-word neighborhood pilot passed four updates covering raw
language and summary inputs, coarse codes, native-center and generated-history
word arms. Fixed primary inference is five concept passes plus four parallel
native-word refinements, without AR fallback. Frozen originals/books/caches and
exact checkpoint/generator reload passed; peak allocation8,656,379,904bytes.
This four-update pilot is systems evidence only; held-out student quality and
throughput require separate measurement from the fixed trained checkpoint.

The independently scheduled teacher pipeline's128first drafts scored
111/128whole-summary passes. After one critique-informed
revision per draft, 123/128passed(96.09375%),
with128valid first judgments and Wilson95%interval
[91.18130%,98.32014%].
Fifteen exact first judgments were reused;113new requests were made. Terra both
critiqued and audited; this is not independent human gold or student accuracy.
All raw requests and first decisions were replayed during finalization.

The original joint teacher gate still **failed**: one final text exceeded the
128stored-token cap and two exceeded128native tokens including EOS. The latter
counts overlap. The stricter all-case capacity rule is not repaired by the
96.09%whole-summary point estimate. No original full-population training
admission is inferred. A separately preregistered143-label exploratory study
uses111individually accepted first drafts plus32older accepted labels, without
choosing the better draft/revision or relabeling this failed gate.

Aggregate `data/experiments/ncm_paper_regime/20261023_native_query_and_label_controls_results_v1/campaign_results.json`, SHA256`cd7848d6a1b258c8f1a623d00b400011ec33c5353d4e4c57d031e956b686bc48`. Registry records retain
confirmed=false; no95%student, compression or speedup claim. Original64development
cases stay the student gate and final256cases remain sealed. The short768-token
prompt prototype cannot establish the full6300-token application range.
