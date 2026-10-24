# Additional10000TRAIN sources: CPU tokenization and alignment

2026-10-24. Hypothesis `ncm-additional10000-source-tokenization-v1`.
Prepare only source metadata while the existing GPU student and bulk teacher
jobs continue. No GPU is initialized or body/model weights loaded. No new teacher
label outputs or selected final sources are read or hashed. The original TRAIN
geometry reference includes historical target columns; the parity check projects
only its source/geometry fields and never passes those targets to tokenization.

Use ALL10000unchanged reserved public TRAIN sources (SHA3e07a05b...), preserving
every source/span/identity/component and original order. Require completed source
materialization, reservation registration and existing whole-corpus exclusions.
Reject all target, summary, word-ID and native-code fields before tokenizer use.

Reuse the existing native Llama tokenizer/chat system, explicit23Oct2026date,
GPT2byte-BPE source tokenization, complete-source round trip and exact character
overlap alignment from the original frozen-body feature pipeline. Require exact
context IDs, GPT2source IDs, alignment weights/hashes, prompt hash and source frame
parity on the original first/longest TRAIN examples. No tokenizer retraining,
source clipping, source-position removal or prompt/date change. Keep all3072
feature channels and original28-layer frozen body identity for future extraction;
the current experiment computes no feature values and chooses no feature dtype.

A metadata-only five-case systems pilot uses one fixed-hash source from each
of the first four bins and the actual metadata-longest source in bin4. Reuse all
five exact prepared rows when processing the full10000. Write each complete
source/prepared row separately and a compact index with hashes, token counts
and alignment/prompt fingerprints. Report actual total GPT2source positions,
maximum prompt/source lengths and exact raw FP32/BF16disk-size projections.
These projections are not model memory or throughput measurements.

Freeze source/plan/tests/auditor before producing data, record actual child and
whole-tool exits, then independently rebuild and compare ALL10000tokenizations
and alignments. Preserve failures; no replacement cases or silent truncation.
This preparation does not alter old FP32feature files, the currently running
native655model, its pending DEV, teacher generation or canonical firsts. It
cannot establish95%quality or speed. Selected final320remain closed/unhashed.
