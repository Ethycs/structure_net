# Frozen Llama features for the fresh NCM development sources

Date: 2026-10-24. Prospective data preparation study
`ncm-clean-development-frozen-features-v1`.

Prepare complete prompt metadata and original frozen Llama source features for
all 64 short and 64 long development sources from the registered exchange
partition study. These features permit the next fixed student to undergo semantic
evaluation on fresh source components. This study measures data integrity and
extraction fidelity, with no model training or summary quality selection.

The source reservation must finish its pilot, full construction, independent
audit and exact registry readback before this workflow starts. Read only its
open development source exports and metadata. Neither old sealed source files
nor the newly reserved final source files may be opened or rehashed. Preserve
all 128 cases, their exact text, public spans and component identities.

Use the existing native 40–60 word, at most 128 token summary system prompt and
literal passage delimiters. Native chat-template tokens must match independent
offset tokenization. GPT2 source tokens must decode to the exact full passage.
Map each source token to overlapping native token character spans with the
existing normalized overlap weights. Round source frames up to multiples of 256,
with an 8192 maximum. Capacity or alignment failure stops the study; no source
clipping, filtering, relabeling or quota reduction is permitted.

The parent remains the original 28-layer Llama 3.2 3B Instruct body, parameter
hash `75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c`.
Run its full source prompt once in BF16, with all original weights and MLPs
frozen. Store detached aligned source features in FP32, shape
`(complete_GPT2_source_tokens, 3072)`. No labels or model outputs enter extraction.

Before any GPU extraction, run focused CPU contract tests and real tokenizer
alignment tests, then complete all 128 metadata records. The CUDA pilot contains
the first short source and the longest native prompt among the other cases.
Extract each twice and require exact raw and aligned feature equality. Reload
and verify saved FP32 tensors. Reuse those exact pilot files in the full cache;
extract the remaining 126 cases after a fresh parent load. Both workers must
verify the unchanged original body and absence of parameter gradients. Peak
allocated memory must not exceed 8.6 GB.

Use physical GPU 0 alone, after the already running dense matched forecaster
supervisor has actually terminated with all seven stages at exit zero and its
campaign registered. Capture its PID and Linux start ticks while live, poll that
same identity, and never restart it. The workflow may prepare CPU metadata while
waiting. A failed upstream lifecycle stops this workflow and preserves artifacts.
An upstream concept-fit miss does not invalidate source feature extraction.

Freeze the source, this plan, tokenizer and parent checkpoint identities, input
campaign hashes and configuration before launching. Use fresh dated roots under
`data/experiments/ncm_paper_regime/20261024_clean_development_*_v1`.
The reproduction entry point is
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:. data/experiments/moba_paper/environment/bin/python -m experiments.structure_net.ncm_clean_development_features workflow`.

The independent CPU auditor replays all public sources, canonical tokenization,
alignment weights, source frames, fixed pilot selection, file hashes and every
saved tensor's identity, shape, finiteness, FP32 precision and detached state.
It checks actual successful stage receipts and the original parent hash reported
by both frozen numerical workers, then registers the complete cache. A successful
classification establishes feature readiness only. The 95% complete-summary
objective still requires the unchanged semantic gates and sealed final evaluation.
