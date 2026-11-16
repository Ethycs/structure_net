# Local response-distillation corpus inventory

The local prepared catalogs contain **208,631 exact source–summary pairs**:
162,353 XSum references, 43,894 recovered conversational pairs and 2,384 reviewed
anchors. Their combined index is available, but it was explicitly **not admitted
as a new combined training corpus**. A larger archive export contains 259,636
length-fitting candidates, including the recovered application subset; these
counts must not be added together.

This is a metadata-only readback for [NAL-STD-DISTILLATION v1](../../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md).
The inventory command verifies seven small historical result-file hashes and
checks that the named payload files still exist with their current byte sizes.
It does not reread the corpus payloads, recompute their historical row counts,
or certify their stored payload digests anew. No examples, final 512 tasks or
sealed audit contents were opened, and no tokenizer, teacher or judge ran.

| Population | Historical pairs | Local payload, below `data/experiments/ncm_paper_regime/` | Actual file bytes | Admission/provenance |
| --- | ---: | --- | ---: | --- |
| Public XSum TRAIN | 162,353 | `20261026_public_training_catalog_v1/training.jsonl` | 705,566,539 | Individually admitted for old public-reference conditional pretraining; not factual gold |
| Recovered conversations | 43,894 | `20261028_recovered_paraphrase_catalog_v1/pairs.jsonl` | 176,330,658 | Noisy paraphrase population; source/context joins and old training-input admission exist |
| Reviewed anchors | 2,384 | `20261027_expanded_teacher_catalog_prepare_v3/cases.json` | 30,242,727 | Previously reviewed training catalog; review is not individual semantic gold |
| Prepared union index | 208,631 | `20261028_combined_prepared_pair_inventory_v1/pairs_index.jsonl` | 192,066,286 | Original record pointers preserved; union-wide near-duplicate split audit incomplete |
| Shared archive candidates | 259,636 | `20261027_available_pair_total_v3/candidate_pairs_unsplit.jsonl` | 697,147,921 | Unassigned split, not training-admitted; fixture/source-group/label checks still required |

The union contains 202,340 distinct exact source texts. There are no duplicate
exact source/target pairs; 881 source texts appear in both the recovered and
anchor populations with different targets. The recovered catalog records 8,055
source components, 38,484 distinct source texts and 40,890 role/date-conditioned
sources. A conversation/component, not merely an individual text or variant
summary, must determine the split. The XSum admission retains the original
`EdinburghNLP/xsum` revision `7d4d486c2f8ef850b1a11aead99b894ff3dd7da9`, article
IDs and original row numbers; related-news-story independence was not established.
Its 320 historical public diagnostic articles remain exposed development data.

The accessible archive census recovered 1,069,574 ordinary source–summary pairs,
931,466 after excluding exact copies. The 259,636 length-fitting candidates
comprise 244,587 raw-summary and 15,049 child-summary merge pairs. The earlier
253,366 atomic and 13,843 merge cache-file counts describe file occurrences,
not independent usable pairs. The approximately 109 GB storage estimate included
overlapping compressed and expanded representations.

Token counts must retain their original units:

- Archive candidates total **106,151,047 input and 15,340,692 target tokens** in
  `cl100k_base`, counting repeated inputs once per pair. These are corpus content
  counts, not observed Qwen training exposure or independent tokens.
- XSum TRAIN has **88,780,546 source `cl100k_base` tokens** and **4,320,525 old
  codec-native lexical target tokens**. Those two numbers use different
  tokenizers and must not be combined as Qwen supervised tokens.
- The historical content filters allow raw inputs 192–2,048 tokens, merges up to
  2,048 tokens and summaries 1–128 tokens. They exclude prompt/template overhead.
  **No retained count under Qwen's validated 416-token total envelope has been
  measured here.** Retokenization and exact assistant-boundary verification are
  prerequisites; oversize examples must be rejected or explicitly segmented,
  never silently truncated.

The existing NCM feature/code caches are tied to their original body, codec and
tokenizer. They are useful provenance evidence, not Qwen full-vocabulary teacher
targets. The prepared text is summarization material; this inventory does not
establish a ready general instruction/language recovery corpus or certify the
numeric/entity/date fidelity of every label.

The narrow next data step is a frozen Qwen-tokenized sample from admitted TRAIN
sources, preserving source-component IDs and exclusions, then a source-group
split and label review appropriate to the new study. Native teacher scoring
must cover the complete selected assistant response and observed EOS. The new
response-data module supports that contract, but no such Qwen sample is claimed
to be prepared by this inventory.

Reproduce the metadata inventory without scanning payloads:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache \
  pixi run --locked --offline -e ncm python -m \
  experiments.structure_net.distillation_corpus inventory
```

For an explicitly supplied, already tokenized response JSONL, `prepare --records
PATH --output NEW_DIRECTORY --max-length 416` writes a bounded pointer-only index
and manifest. It checks source groups, exact token duplicates, causal/EOS record
contracts and lengths, retains duplicate origin pointers, and activates the
first exact pair per split. It creates no new scientific admission and copies
no source/response payload. The default 10,000-record bound fails rather than
truncating. A study must still declare near-duplicate policy and benchmark
exclusions; token-exact equality is not semantic deduplication.

Evidence checked in this readback:

| Receipt | SHA-256 |
| --- | --- |
| Combined union `20261028_combined_prepared_pair_inventory_v1/result.json` | `a10aacce5d41a80ecc9fa1119b3623d527a43d8e01e99d2de9467517356f35cd` |
| Public admission `20261026_public_training_catalog_audit_v1/result.json` | `63045e80261a03a7bdf75aec961475c21fdc6d4debeb6c801ab6df3423396aa5` |
| Recovered catalog `20261028_recovered_paraphrase_catalog_v1/result.json` | `ef2ba5661a46498510b417f3aef56d36ab6530f51190837dce46e849efce5c5c` |
| Recovered input admission `20261028_recovered_paraphrase_training_prepare_v1/result.json` | `1c979cf4769b8363d146687a809b10ae77a47b83b98eaed35067cdb189d988e7` |
| Archive census `20261027_available_pair_total_v3/result.json` | `443b7011c80f42a250d4c6d729284f05f87be8a46a82676772d38495ce613b64` |

The command emits the remaining preparation hashes, original payload digests,
paths and explicit `payload_rehashed_now:false` flags. This is corpus availability
evidence, not recovered model quality or a new training launch.
