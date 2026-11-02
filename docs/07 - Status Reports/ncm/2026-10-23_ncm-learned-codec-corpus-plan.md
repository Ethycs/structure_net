# Prepare source disjoint text for a learned NCM codec

Date: 2026-10-23. Exploratory preparation for
`ncm-broad-learned-codec-reconstruction-v1`. The clean native teacher is still
being evaluated. Its result does not determine this representation dataset.
No codec training or summary-quality claim is admitted by this preparation.

The released NCM uses a learned codec and text decoder. The current native
hybrid instead clusters fixed token embeddings and uses a causal word readout.
The released codec previously recovered 695/717 tokens on 20 novel short texts;
the small adapted codec recovered 664/717. Both receive correct target codes.
Broader codec reconstruction must be tested before another concept-generator
transfer is attributed to NCM. Preserve those failures.

Reuse the released GPT-2 text tokenizer and 48-position codec architecture.
All 252 accepted complete-summary targets fit 192 GPT-2 positions including
EOS; the longest needs 136. A later reconstruction control can represent each
complete target as four independent 48-position codec blocks, concatenate raw
predicted token IDs, then decode once. This blocking is an explicit adaptation,
not an unchanged published generation configuration. Never trim a summary to
48 or 128 GPT-2 tokens. The parent model may keep its native tokenizer.

For unpaired codec text, use only groups in the existing 1,024 raw-language
TRAIN cohort, removing every group in the 252 accepted-summary cohort. Check
exclusion by session, group and exact source hash against original development,
raw-language diagnostics, new longer-source diagnostics and new sealed-final
metadata. Inherit original final exclusion from the verified original train
partition; do not open either final text cohort.

Read only `native_train.jsonl` from the restored shared data. Verify its hash
against the original manifest and require every row to retain split=train.
For allowed groups, take full raw sources and saved atomic target text as
unpaired reconstruction material. Atomic targets are not complete-summary
supervision. Deduplicate exact text, preserve all origins and exclude exact
accepted-summary target strings. Reject embedded EOS/PAD token IDs explicitly.
Tokenize each full text, append EOS once at its true end, and pack consecutive
48-token windows. Keep every tail with PAD; do not append EOS at artificial
window boundaries. Preserve token arrays and per-window origin/index metadata.

Keep the 252 accepted complete summaries outside codec fitting, with their full
target text plus EOS packed into 192 positions. Verify exact text round trip,
no truncation, no fit-group/session/source-hash overlap and complete ID coverage.
These remain training labels for the later conditional summarizer; they are
held out only from codec reconstruction fitting. They are not a final task test.

Focused CPU tests cover lossless token packing, EOS placement across boundaries,
capacity rejection and exact reconstruction of the valid flattened token stream.
Save a fresh `20261023_learned_codec_corpus_v1` root with source/config/input
hashes, counts, all selected origins, strict JSON metadata and token tensors.
Changing selection or packing requires a new root. A separate protocol and
real CUDA pilot must precede codec weight updates. Original parent weights
remain frozen. Codec reconstruction, teacher-forced recovery and an oracle
target-code control cannot establish 95% source-to-summary task accuracy.
