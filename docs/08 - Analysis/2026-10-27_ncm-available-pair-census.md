# Available source–summary pair census

The completed accessible-archive census recovers **1,069,574 distinct ordinary
source–summary pairs**, of which **931,466 are non-copy pairs** and **259,636
non-copy candidates fit the application's content-length limits**. The latter
is 108.9 times the 2,384 prepared pairs. It is a measured candidate supply before
fixture removal, source-disjoint splitting, native-tokenizer checks and semantic
label review; it is not a training-admitted count or model accuracy.

All contributing workers completed successfully. The final status is
`full_shared_pair_inventory_reconciled_and_exported`; a separate readback reports
`export_readback_and_length_partitions_verified`.

| Input → stored target | Distinct pairs | Exact source copies | Non-copy pairs | Length-fitting candidates | Distinct inputs among fitting candidates |
|---|---:|---:|---:|---:|---:|
| Raw text → summary | 792,176 | 138,108 | 654,068 | 244,587 | 173,801 |
| Ordered child summaries → parent summary | 277,398 | 0 | 277,398 | 15,049 | 14,239 |
| Total | **1,069,574** | **138,108** | **931,466** | **259,636** | — |

Pair identity includes the input kind and exact source and target content hashes;
source namespace aliases do not create extra pairs. Different summaries of the
same input remain different pairs. Distinct input texts are not necessarily
independent conversations or nonoverlapping source spans.

Raw inputs must contain 192–2,048 `cl100k_base` tokens. Merge inputs must be at
most 2,048 tokens. All retained targets contain 1–128 tokens. These are content
lengths, not complete role/date-conditioned prompt lengths. Raw batches and their
generation caps are not counted as individual summaries.

The independent merge audit finds three one-child identity copies, 222 targets
that copy one of multiple children, eight exact child concatenations, and 14,816
other targets. Excluding the three one-child identities leaves **259,633** fitting
candidates. These mechanical categories do not certify summary quality.

## Why the total is larger than the original cache count

The census accounts for all **798 published bundles and 2,357,622 files**, the 500
original stores, and six original source-package files. It inspects the JSON
records omitted by the sender's filename index, streams all 67 large JSON files,
restores unindexed SQLite views and WAL companions, and resolves pointer-, hash-,
request- and batch-job-bound targets against real source text.

| Recovery stage | Fitting ordinary pairs after global union |
|---|---:|
| Indexed stores, caches and request/response receipts | 127,468 |
| Additional body stores and source-pointer labels | 224,826 |
| Deeply embedded JSON and remaining database views | 249,792 |
| Prompt-keyed hierarchy batches | **259,636** |

The original 253,366 atomic and 13,843 merge cache files are file occurrences,
not the whole corpus. Their content hashes deduplicate to 229,673 and 13,326
respectively, and additional stored pairs occur in the other representations.

The archive has 28,571,676,558 compressed payload bytes and 79,421,060,633 bytes
of archived file payloads. Those representations overlap. The earlier approximate
109 GB storage figure is therefore not 109 GB of independent labeled text.
The published catalog also lists 804 inaccessible paths across 80 source roots;
unpublished content is outside this measured census. Two invalid JSON files are
reported rather than silently accepted.

## Other recovered data and unresolved records

The four-stage union contains a further **141,551 raw-to-structural-summary
conversions** and **269 image-context-dependent LOC0MO pairs**. These are separate
from the ordinary total: reconstructing raw spans for a structural target does
not establish its original model input, and the text export does not reproduce
LOC0MO's image context. Only 55 and 170 respectively meet the same content-length
test. Including these special classes gives **1,211,394 distinct recovered pairs
across all input kinds**, not 1,211,394 ordinary text-summary training examples.

Among ordinary pairs, the disjoint exclusions are 138,108 exact copies, 261,060
non-copy raw inputs below the summarization threshold, 1,152 inputs over 2,048
tokens, and 409,618 targets over 128 tokens. Longer targets could inform a later
auxiliary task or be relabeled; they are not silently truncated into this supply.

Residual unresolved records include 399 distinct atomic-cache contents and
12,660 merge-cache contents, ten source-pointer labels, 172 source-hash labels,
and 536 structural labels with missing or ambiguous children. These categories
can overlap each other and pairs already recovered elsewhere. They must not be
added together as extra unique pairs. Missing inputs are not invented by matching
target text against other target-only records.

## Inspectable export and validation

The actual source and target text for all **259,636** fitting ordinary candidates
is exported to:

`data/experiments/ncm_paper_regime/20261027_available_pair_total_v3/candidate_pairs_unsplit.jsonl`

The file is 697,147,921 bytes. Its SHA-256 is
`38ff6ca93b5d010c3879d5e18411865299c90be7f41f493a51faebc31fdf6960`.
Every row has `split: unassigned` and `training_admitted: false`, a content pair
ID, input kind, source/target text, token counts and representative provenance.
The normalized export is not the final role-conditioned training prompt format;
role, date and spine metadata remain in the linked source artifacts.

Across these candidate rows, inputs total **106,151,047 tokens** and targets
**15,340,692 tokens**, or **121,491,739 tokens combined**. This sums pair rows;
repeated inputs with alternative targets count repeatedly. It is neither a
unique-token count nor measured training exposure.

The exporter checks every source/target hash against the counted database. An
independent pass reads the entire JSONL back, verifies the export hash and count,
checks all length/admission flags, reproduces the per-kind counts, and audits
trivial merges. The SQL count also reconciles independent disjoint exclusion
buckets. Zero identical source/target content pairs occur in both ordinary input
kinds. Parser/join tests cover protected source fields, exact Unicode text,
oversized records, malformed input, verified spans, ambiguous children and the
batch-job prompt protocol. No new teacher or judge calls were made by the census.

The final result is
`data/experiments/ncm_paper_regime/20261027_available_pair_total_v3/result.json`,
SHA-256 `443b7011c80f42a250d4c6d729284f05f87be8a46a82676772d38495ce613b64`.
Its `inventory_audit.json`, `candidate_export.json`, stage databases and source
result bindings retain the reconciliation. The native scan, source join and
final union workers all have observed `ExecMainStatus=0`, `SubState=exited`, and
zero restarts. Manually superseded parser workers are recorded separately and
are not represented as successful completions.

The next data gate is provenance-based fixture filtering and source-group
splitting, including exclusion of protected evaluation sources, followed by
native-codec capacity checks and summary-quality review. This inventory establishes
the supply to do that work; the 2,384 prepared examples did not establish scarcity.
