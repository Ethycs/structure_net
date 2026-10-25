# Audit historical TRAIN summaries matched to existing frozen features

The new non-LongMemEval engineering pool remains a separate partition with
732 TRAIN and 156 test candidates. It excludes the whole core collection.
Separately, a metadata-only join found 4,949 existing core TRAIN summary targets
whose exact source hashes match 4,949 already prepared, cached TRAIN components.
This protocol audits that historical pool for a larger supervised experiment.
It makes no claim that the whole LongMemEval benchmark is untouched: historical
checkpoints already saw part of it. Existing source-disjoint DEV and sealed final
cohorts, plus all teacher/component controls, remain excluded.

Preserve the first original physical TRAIN record per exact source hash, as
fixed by the completed metadata join. Independently replay raw source/target
hashes, original IDs/groups/session bindings, original cl100k counts, full native
BERT frames and every original range/protected-metadata exclusion. Require the
exact source bytes to match their cached source metadata, canonical component,
prompt/alignment hashes and lossless feature-header binding. Protect all 512
reserved components and compare against all 192 open DEV/component source texts.
Never read or hash sealed final source files or original old-test text files.

Use a new CPU-only root and keep every conflict/exclusion. No alternative target
selection, target repair, clipping, new model calls, student training or accuracy
claim. Existing stored targets remain unreviewed; syntax/length/source bindings
do not certify their semantic quality. Bind the already completed 10,000-source
cache manifest and its audited provenance; verify each feature payload at use
through the existing bounded loader. No bulk re-extraction or lossy storage.

Export source/target/code-frame bindings and the corresponding source and feature
subsets. Only a separately frozen training protocol may admit these outputs.
The original 128 DEV and sealed 320 final scope remains the quality target;
engineering test data is additional evidence, not a replacement for that scope.
