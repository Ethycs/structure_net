# Decoder repair reconstructs all 732 training summaries

The staged decoder-only experiment passed its fixed reconstruction checks. Given
reference codes, the repaired decoder reproduces all 732 supplied engineering
TRAIN summaries in the native tokenizer's rendering: 30,109/30,109 active word
IDs match. This is a representation result, not held-out summarization accuracy.

| Reconstruction control | Before | After |
| --- | ---: | ---: |
| Exact rendered summaries on the fixed 16 cases | 5/16 | 16/16 |
| Active word-ID agreement on those cases | 650/672 | 672/672 |
| Full732 reference-code reconstruction | Not measured in this run before training | 732/732 summaries; 30,109/30,109 word IDs |

The earlier errors in `tensor products`, `dependency-aware`, `manual pinning`,
and `five unmerged commits` are repaired. The native BERT tokenizer lowercases
text and spaces some punctuation, including code identifiers. Therefore exact
rendering means equality after applying the same native tokenizer to the target;
it is not byte-for-byte recovery of capitalization or original formatting.

Only 18,870,656 text-decoder parameters changed. The source-to-code generator,
source attention, codec encoder, codebooks and Llama source features stayed
frozen. All protected parameters and buffers were hash-checked. The generator's
source-only codes remained identical before and after decoder training. With
the full-data decoder, those codes render all 16 control targets correctly.

The control used 256 updates, four examples per update, and 64 exposures per
case. Its optimizer loop took 54.25 seconds. The full run reset to the same
original decoder, used 1,024 updates and 4,096 example presentations across 732
cases, giving each case five or six exposures. Its optimizer loop took 1,338.38
seconds, or 22.31 minutes. The final 16-update mean training loss was 0.000568.
Both stages used FP32 weights, FP16 training and FP32 evaluation on GPU2, the
RTX2060SUPER. Full-run peak allocated GPU memory was 1,529,235,968 bytes. These
timings and memory figures concern decoder training, not inference throughput.

Four focused CPU tests passed, followed by a two-update CUDA lifecycle pilot.
Both fixed training stages exited zero. Fresh-parent reconstruction of the
decoder checkpoint reproduced weights, optimizer/scaler/RNG state and source-only
outputs exactly. An independent arithmetic audit replayed all output counts,
training steps, schedules and exposure counts. No intermediate checkpoint was
selected for quality. No held-out labels or sealed final sources were used.

The full-data reconstruction gate passed. Source-to-code training on the 732
cases and independent semantic evaluation remain separate stages. The original
128 development sources have no identifier, session, group, source-hash or source
component overlap with the 732 training cases. Both previously qualified judges,
Sol and Terra, responded to synthetic readiness probes; their qualification
records and all 3,539 inherited first judgments verified. These probes are not
summary quality results.

The user subsequently selected the 6GiB RTX2060, physical GPU1, for the next
stage. Existing Triton services occupy part of that card. CPU optimizer-state
offload passed eight CPU tests and a separate actual RTX2060 AMP contract:
four updates differed from GPU AdamW by at most 3.7253e-9 in weights, and optimizer
state reloaded exactly. Full NCM training memory fit still requires its own
representative pilot. There is no GTX1070 or RTX2070 installed on this host.

Plan: [decoder repair protocol](../07%20-%20Status%20Reports/2026-10-25_ncm-sequential-decoder-plan.md).
Implementation commit: `80f4288`. RTX2060 migration commit: `1a222ef`.

The decoder checkpoint is
`data/experiments/ncm_paper_regime/20261025_sequential_decoder_train_v1/model.pt`,
SHA256 `9d5525b8a101954b7d4e1c3f3c920668b930118e8d2c1365e17f7fb2b19d0083`.
It is compositional: load the sequential parent checkpoint identified in its
metadata, then load its decoder tensors. It includes FP32 optimizer state.

Independent aggregates are
`data/experiments/ncm_paper_regime/20261025_sequential_decoder_capacity_audit_v1/campaign_results.json`
and
`data/experiments/ncm_paper_regime/20261025_sequential_decoder_train_audit_v1/campaign_results.json`.
Raw outputs, losses, frozen source snapshots, fingerprints and actual terminal
receipts remain in the corresponding run directories.

The 95% held-out task-accuracy target and inference speedup are not established
by this decoder reconstruction experiment.
