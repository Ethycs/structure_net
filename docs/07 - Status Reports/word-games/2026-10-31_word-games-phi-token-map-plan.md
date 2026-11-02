# WORD_GAMES: tokenizer map between Phi-4-mini and Phi-4 14B

October 31, 2026. Systems compatibility experiment; no training or quality claim.
Hypothesis: `word-games-phi-token-map-v1`.

Build bidirectional maps from the exact cached GGUF vocabularies used in the
previous smaller-Phi speculative experiment. Use vocabulary-only loading on
CPU in the existing pinned serving image and its Pixi Python runtime. Do not
load weights onto GPUs or change serving endpoints. The controller runs from
the repository root through the locked word-games Pixi environment.

Compare decoded token bytes, not integer IDs or display strings. Exclude
control, unknown, unused and zero-byte tokens from ordinary-text mapping.
Never equate chat boundaries, BOS/EOS, padding or role markers by their text.
For each ordinary token, record an exact-byte single-token counterpart if one
exists, plus a canonical destination-token decomposition of that token's bytes.
Verify every stored sequence by byte reconstruction; unavailable decompositions
must be explicit. Bind maps to both checkpoint hashes and tokenizer metadata.

Report whole-vocabulary coverage in both directions and weighted coverage on
the frozen 36 WORD_GAMES requests and saved target responses. Test Unicode,
partial UTF-8 byte tokens, whitespace, JSON/numbers, duplicate pieces and
control tokens. Compare concatenated per-token maps against whole-span native
retokenization: byte equality and canonical token-ID equality are distinct.

Make the map available as an opt-in speculative adapter. Mapped proposal IDs
are usable only with full target verification. A byte-preserving map does not
guarantee identical token boundaries, interchangeable embeddings/logits, or
the same draft distribution. The default adapter and deployed runtime remain
unchanged. Preserve model-specific chat templates. If mapped conditioning is
offered, label it as potentially noncanonical and experimental.

Pass the compatibility gate only if all stored ordinary-token decompositions
reconstruct exact source bytes, special tokens are blocked from ordinary maps,
saved maps reload with integrity checks, and all corpus/fuzz byte checks pass.
Record conversion timings separately from inference: this experiment cannot
establish an end-to-end tok/s gain, model accuracy, or the prior 40-tok/s goal.
