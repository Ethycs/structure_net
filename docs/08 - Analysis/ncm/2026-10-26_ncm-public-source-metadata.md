# Every admitted public source reproduces its original prompt and alignment

All **162,353** admitted public training sources passed exhaustive CPU replay
of their native Llama prompt IDs, GPT2 source IDs and character-overlap
alignment. The complete saved records matched a separate execution of the
original preparation function. This admits the metadata for subsequent
feature extraction; it provides no model-quality evidence. The latest
completed application experiment remains at 0/128 semantic passes per arm.

Study `ncm-public-source-metadata-v1` is classified
`all_public_full_source_metadata_replayed`. Its authoritative audited result
is `data/experiments/ncm_paper_regime/20261026_public_source_metadata_full_audit_v1/result.json`,
SHA-256 `a2cc0cdd45d7d7d3967e68f98bbb71ab03df7f2d2b5fa7c3218ae7af67a382b4`.
The [preparation plan](../07%20-%20Status%20Reports/2026-10-26_ncm-public-source-metadata-plan.md)
and [audit plan](../07%20-%20Status%20Reports/2026-10-26_ncm-public-source-metadata-audit-plan.md)
were fixed before their respective executions.

| Measured quantity | Result |
| --- | ---: |
| Complete source records | 162,353 |
| GPT2 source positions | 91,344,717 |
| Native Llama prompt positions | 125,616,400 |
| Maximum source positions | 4,124 |
| Maximum native prompt positions | 2,275 |
| Exact repeated public pilot records | 9/9 |
| Exact original TRAIN geometry controls | 2/2 |
| Estimated full FP32 feature bytes | 1,122,443,882,496 |

The original system prompt, pinned date 23 Oct 2026, source strings and every
source position were retained. An explicit source-field projection removes
reference summaries, native target IDs and codes before tokenization.
The audit checks complete original-order byte indexes, per-record hashes,
source identities, valid prompt positions and finite normalized overlap
weights. It repeats tokenization for every row, not a sample. Its tokenizer
and alignment implementation is shared with the producer; the independence
is execution and binding verification, not a separately implemented tokenizer.

The first nine-source pilot covered the original first row and distinct
nearest source lengths across the admitted range. Full preparation reproduced
all nine records. The audit first passed on those nine records, then processed
the entire population with four spawned CPU workers, 128 records per task and
at most eight pending tasks. The ordered disk dataset retains index metadata
in memory and loads source records on demand; it does not load corpus-wide
feature tensors. Fourteen focused CPU tests passed for the dataset, producer
boundary and replay checks, plus one registry-boundary test.

Full preparation took 1,690.68 seconds within its recorded processing scope.
Exhaustive audit took 481.92 seconds within its recorded scope. The reported
parent and maximum individual-worker peak RSS are each about 1.26 GB; these
are not a measurement of combined process memory. Actual exit-zero tool
receipts are retained for pilot, full preparation and both audits.

The measured source maximum is substantially longer than the 2,257 positions
covered by the prior feature pilot, so full-range GPU qualification is still
required. The FP32 estimate exceeds the root filesystem's free space.
Inspection also found about 3 TiB free on the writable `/data2` volume, allowing
a complete disk-backed cache instead of repeated body computation. The
separate feature-store plan qualifies that storage and extraction path.

No model weights were loaded, no features or gradients were computed, and no
new judge requests occurred in this metadata study. Public references remain
unverified pretraining labels under the existing catalog admission. Reserved
public development articles and application split protections are unchanged;
the sealed final320 remains closed. The 95% complete-summary goal is unmet.
