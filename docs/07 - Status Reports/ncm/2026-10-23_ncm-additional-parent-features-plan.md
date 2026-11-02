# Cache unchanged-parent source features for the additional training sources

Date: 2026-10-23. Preparation for the full-hierarchy NCM experiment.
The source-root capacity control passed on training cases; the latest complete
summarizer still scores0/128. Feature preparation is not task accuracy.

Use all512fixed, public, source-disjoint inputs from the additional whole-summary
source selection. Read no teacher drafts, critiques, accepted labels or outputs.
Verify their complete public text, case/session/group/hash identity and exclusion
from the existing252TRAINcases,128development cases and sealed cohort metadata.
No final source or output file is opened. Cache even sources whose eventual
teacher label may fail; label selection happens separately.

Use the same canonical SYSTEM/chat prompt, original Llama3.2-3B-Instruct tokenizer
and complete28-layer frozen body as the earlier source cache. Preserve all tokens
with no source clipping. Build exact native offsets and GPT-2 source offsets,
and reuse the tested character-overlap alignment in `ncm_word_final_data`.
Require prompt-token identity, exact GPT-2 source round trip, positive normalized
coverage and complete frames in multiples of256up to8192. Export source metadata
only, with no target words or concept codes. Record observed maximum lengths.

Run the original body in BF16 eval/no-grad on physicalGPU0. The body hash
75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c must match
before and after. Original weights and MLPs remain frozen. Align/pool inFP32
and retain FP32 per-GPT-2-token vectors. TF32is disabled.

First use a real two-case feature pilot: first selected source and the longest
native prompt among all512. Require exact repeated native/aligned features,
finite complete shapes, no gradients, unchanged body hash and peak allocation
below8.6GB. Only after observed exit0cache the full512, retaining the two pilot
entries exactly. Reuse the existing feature-cache implementation unchanged
behind an explicit new-root/config wrapper. No teacher label is an input.

Fresh roots: `20261023_additional_parent_{data,pilot,features}_v1` under the
paper-regime directory. Bind all producing source/config/input hashes and record
actual terminal receipts. This preparation can run concurrently with the
CPU/network label workflow; keep one own model per physicalGPU. Student training
requires separately vetted labels, a fixed full-hierarchy protocol and a real
maximum-frame gradient/reload pilot. Both final cohorts remain closed. No
summary accuracy, compression or speed claim follows from this cache.
