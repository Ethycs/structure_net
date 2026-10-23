# Isolated quality training: data and maximum-frame controls passed

Date: 2026-10-23. Classification
`mixed_length_cache_and_isolated_word_training_controls_passed`.
Aggregate `data/experiments/ncm_paper_regime/20261023_isolated_quality_controls_results_v1/campaign_results.json`, SHA256 `1317ee9020f5cdf78dd38584e737e807eb976a8a2f269830d600422e6b479f82`.

All 252 vetted complete-passage training summaries passed source identity,
provenance, target round-trip and128-token capacity checks:143old short labels
and109longer-source labels. Saved atom snippets were not silently treated as
complete-passage summaries. Longer training sources span512–2048cl100k tokens;
their longest native chat prompt contains2,276tokens and uses a2,304-token frame.

The split audit found no ID/session/group/exact-source-hash overlap with
development or diagnostic/new-final metadata. All252identities match the
original training partition. Original256final ID/hash metadata was derived
from the earlier public proof, inheriting session/group exclusion from the
original partition; its source texts were not opened. Neither were the newer
64sealed source texts. Full selected sources and targets were not clipped.

The first cache pilot failed a systems check: token IDs were passed where
the native query method requires embeddings. Its exit-one receipt, source,
failure record and partial files remain intact. A tiny actual Llama CPU test
caught the corrected contract. The v2 pilot recomputed the longest cache and
passed exact native-query save/reload checks. The full v2 cache covers all252
cases, retaining143old cache files and computing109new ones. Original source
backbone weights remained unchanged.

The separate real gradient pilot performed two clean word updates, one on
the first short case and one on the absolute longest selected source. Before
training, the zero word bank matched cached parent logits exactly on both.
Only4,587,520new word Q/K/V/O parameters received gradients. Original MLPs,
embeddings, word head and trained concept bank stayed frozen. Parent/body,
concept predictions, books and CPU cache passed integrity checks, along with
exact parameter/optimizer/RNG/output reload. Peak allocated GPU memory was
8,435,067,904bytes.

These controls admit the fixed512-update clean quality-teacher experiment;
they do not establish that it finished, reached95%, or trained a working NCM
replacement. The last measured converged hybrid remains41/64complete-summary
passes; unchanged3B scored52/64. Full unseen-summary grading after training
is required. Acceleration distillation and all final evaluation remain later
gates. No speed or compression claim follows from these systems results.
