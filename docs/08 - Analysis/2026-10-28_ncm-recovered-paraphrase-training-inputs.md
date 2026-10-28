# Recovered conversational training inputs

The source/target catalog and full role-aware prompt preparation are complete.
They provide 43,894 complete historical paraphrase targets from 38,484 distinct
source texts across 8,055 source components. These are noisy training candidates;
their availability does not establish label correctness or model accuracy.

| Quantity | Count |
|---|---:|
| Guarded candidates examined |104,503|
| Retained source–target pairs |43,894|
| Distinct raw source texts |38,484|
| Distinct source/role/date inputs |40,890|
| Source components |8,055|
| Inputs with assistant role |40,055|
| Inputs with user role |835|
| Inputs without an original timestamp |270|
| Raw source texts with multiple original timestamps |2,286|

The selection retains all eligible non-substring targets with unambiguous
original speaker metadata. It excludes the entire 128-component saved-label/codec
diagnostic sample and duplicate native frames from that sample. All inherited
source guards remain in place; sealed-final text remains unopened. Every target
fits the complete native 128-token frame and the 128-token application limit.
Sources remain within 192–2,048 application tokens. No text was truncated.

Separate indexed files preserve source-only inputs, targets, original context
provenance, and all target variants grouped by raw source. Future training must
sample source texts uniformly and rotate their target variants; 43,894 pairs must
not be reported as 43,894 independent inputs. The catalog preparation examined all
rows and replayed every emitted byte offset and source/target join. It completed
in 143.2 seconds with 626 MB peak host RSS.

The source prompt now exposes the original role and timestamp before the
passage. Neither targets nor summarizer identities enter the prompt. The
chat-template date remains 23 Oct 2026, separate from each original timestamp.
All 40,890 sources were tokenized and aligned without clipping; all seven pilot
records reproduced exactly, and original TRAIN geometry controls remained exact.
The full pass took 618.0 seconds with 821 MB peak host RSS. It contains 19,147,700
aligned source positions, with a maximum 3,464 per input; the longest native Llama
prompt is 2,334 tokens. Full uncompressed FP32 features would occupy 235,286,937,600
bytes on disk, not simultaneously in RAM.

The singleton target-code pilot reproduced all 128 saved reference hierarchies
and 32 new TRAIN controls. It used 783 MB peak allocated GPU memory. Full encoding
is running on GPU2. Singleton calls avoid the batch-sensitive code changes found
in the earlier diagnostic. The frozen-body feature pilot passed all 15 cases,
including exact repeated extractions and unchanged original TRAIN controls.
Its separate disk replay also passed. Peak allocated GPU memory was 6.63 GB;
the pilot's 314 MB of raw features occupied 70.4 MB in the lossless store. That
pilot compression ratio is not a measurement of the full corpus. Full feature
extraction has now launched on GPU0 and is not yet complete.

The requested source hierarchy continues training independently on GPU1. Every
one of its 12 NCM blocks can read source-token leaves and pooled ancestors. The
matched 12-block dense-source arm still must train on the same GPU, data and
exposure. The new corpus and role-aware prompts are not installed in either arm.
No hierarchy quality improvement or inference speedup has yet been established.

There is no new complete application accuracy score. The latest completed DEV
result remains 0/128. The active public epoch4 short cohort has 0 passes among 25
valid judgments and 39 unresolved transport failures; that is not 0/64 measured
accuracy. Codec reconstruction 117/128 remains a separate result. Final320 is
sealed and the 95% task goal remains unmet.

Evidence under `data/experiments/ncm_paper_regime`:

- `20261028_recovered_paraphrase_catalog_v1`: actual33942/`2376df`/0.
- `20261028_recovered_paraphrase_metadata_pilot_v1`: actual98329/`f81653`/0.
- `20261028_recovered_paraphrase_metadata_full_v1`: actual87080/`f90fcb`/0.
- `20261028_recovered_paraphrase_codes_pilot_v1`: actual28084/`3db4b9`/0.
- `20261028_recovered_paraphrase_codes_supervision_v1`: full-code worker launched
  by actual`1ec58e`/0; launch success is not worker completion.
- `20261028_recovered_paraphrase_features_pilot_v1`: extraction actual
  11455/`13f9a7`/0; independent disk audit actual 85167/`6c5a69`/0.
- `20261028_recovered_paraphrase_features_supervision_v1`: full-feature worker
  launched by actual `4b7b8d`/0; launch success is not worker completion.

Focused tests passed: 5 catalog contracts, 6 prompt contracts, 3 target-code
contracts, and 2 feature-selection contracts. Source, tests and plans are frozen
in commits `dd3a8d1`, `dc73f66`, `32d6be2`, and `efab640`. Source/target files are
readable JSONL; tensor payloads are kept under `/data2` with indexed disk access.
