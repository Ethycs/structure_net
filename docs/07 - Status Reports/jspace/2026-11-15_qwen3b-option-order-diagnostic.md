# Qwen3B option-order diagnostic — prospective protocol

Use the already scored 32 WinoGrande cases from the completed fourteen-block
sanity study. This is a diagnostic on exposed cases, not a new holdout. No
training, checkpoint selection, final-confirmation access, or speed measurement
is authorized by this protocol.

Freeze the original native Qwen3B and the accepted compact layers22–35 checkpoint
`814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401`.
Keep the original embedding/readout and the trained compact final RMSNorm.

For every case, preserve the question and option strings, then exchange only
the two option positions and map the gold index `y` to `1-y`. Regenerate both
prompts using the pinned tokenizer, the identical system instruction and chat
template. The original tokens and normalized identity must exactly match the
frozen sanity case. Keep the original maximum384-token input range. Freeze all
64 variant identities/tokens before either model scores them.

Each model uses singleton, unpadded BF16 eager execution and the same choice-row
readout as the sanity study. The native phase performs a fresh full forward,
checks the prior development boundary control bit-for-bit, and saves lossless
BF16 pre-layer22 states. The compact phase reads these exact states and runs
only the accepted14-layer suffix plus its trained norm. Each arm must reproduce
all32 original saved records, including logits, exactly before that arm scores
its32 swapped prompts. Any failure stops that phase without interpreting the
swapped outputs. Both completed phases and their original-replay checks are
required before finalization.

Use one reserved RTX3060 at11264MiB, allocator cap11008MiB, for each GPU phase.
The CPU-only prepare and finalize phases never initialize CUDA. The new module,
tests, this protocol, inherited numerical sources and all relevant input hashes
are frozen in the new versioned root before execution. Existing sources and
results remain unchanged.

Report each model's original and swapped accuracy, average accuracy across the
two variants, both-correct and either-correct counts, and consistency after
mapping the swapped prediction back to the original option text. Report A-label
frequency separately from the frequency of selecting the original option text.
Show paired native-versus-compact gains/losses for each variant. These describe
option-order sensitivity; they do not prove its cause or establish preserved
reasoning ability. `confirmed=false`; the reserved512-case confirmation remains
sealed.

Default root: `data/experiments/jspace_3b_option_order/20261115_v1`.
CLI phases: `prepare`, `native`, `compact`, `finalize`.
