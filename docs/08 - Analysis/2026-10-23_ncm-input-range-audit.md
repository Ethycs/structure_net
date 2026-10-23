# Current NCM data covers short sources only

Date: 2026-10-23. Read-only CPU input-exposure audit; no model quality measured.
Actual native context lengths: `{'language_train': {'count': 1024, 'min': 128, 'median': 128.0, 'max': 128}, 'language_diagnostic': {'count': 64, 'min': 128, 'median': 128.0, 'max': 128}, 'summary_train': {'count': 143, 'min': 421, 'median': 525, 'max': 671}}`.
These include native chat instructions for summary inputs. The language arms
see their stored token windows, not their original complete raw sources.

Summary training source cl100k lengths: `{'count': 143, 'min': 194, 'median': 298, 'max': 444}`.
Development source cl100k lengths: `{'count': 64, 'min': 193, 'median': 310.5, 'max': 437}`.
Approved candidate-source cl100k lengths: `{'count': 8192, 'min': 192, 'median': 298.0, 'max': 454}`.
Summary target cl100k lengths: `{'count': 143, 'min': 23, 'median': 87, 'max': 123}`.
Training/development bypass cases below 192 source tokens:
0/0.
Sources over 512 cl100k tokens in summary training/development/candidates:
0/0/0.

Every summary source ID/session/group/hash and actual native prompt was matched
to its fixed label record. Application units differ: a raw fragment may contain
2,048 cl100k tokens; batches allow eight fragments within a 6,300-token prompt.
Stored summaries allow 128 tokens, while the extractor's 32-word target is a
separate task. No raw batch or hierarchy-merge quality is measured here.

The current run remains unchanged. Passing its short-source 64-case development
gate cannot establish those longer operating ranges. Final source-disjoint,
representative task evaluation remains required; the original 256 sealed sources
were not read. Neither exposure counts nor baseline scores establish 95% student
accuracy, compression or speedup.

Result `data/experiments/ncm_paper_regime/20261023_application_input_range_audit_v1/result.json`, SHA256 `bdec7dcc019a61dbb1431ba0eac556317e7c7f5b8b8ed596c5bb2fc5cbbdd4d8`.
Producer hashes and strict JSON remain append-only.
