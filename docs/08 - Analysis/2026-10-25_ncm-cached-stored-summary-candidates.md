# Stored summaries matched to cached training sources

The existing shared corpus supplies **4,949 unchanged stored summaries matched
byte-for-byte to 4,949 reserved source components with cached frozen-body
features**. This preparation made no teacher requests and changed no running
experiment. It does not qualify the labels semantically or authorize a new
training run by itself.

The completed auxiliary source audit replayed all73,836 metadata-eligible pairs
from the original204,967-row TRAIN file. It found no matches to the128 open DEV
sources or64 open component-test sources under normalized equality or the fixed
32-shared-fivegram/80%-containment rule. Removing7,762 exact source duplicates
left66,074 pairs across15,067 source sessions and14,897 original groups. Existing
metadata exclusions cover512 reserved components. The sealed final files were
not opened; full-text checking against those files remains deferred.

Every one of the4,949 cached-source joins is present in the retained audit output,
with matching original label ID, physical line, source hash and target hash.
Selection retains the first eligible original pair for each exact source hash;
neither model output nor label quality chooses the target. The feature manifest
and source index match the reserved source identity. This join did not replay
feature payloads; the previous full10,000-feature audit remains their provenance,
and any training loader must still validate the payload it reads.

| Complete source tokens | Matched pairs |
| --- | ---: |
| 192–511 | 4,517 |
| 512–767 | 369 |
| 768–1023 | 59 |
| 1024–1535 | 1 |
| 1536–2048 | 3 |

These candidates are heavily weighted toward shorter sources. They cannot alone
establish adequate training coverage for long summaries. They also belong to the
legacy shared LongMemEval corpus, not the broader non-LongMemEval archive requested
later. Historical checkpoint exposure remains part of the record. The earlier
source-audit plan explicitly treats this pool as auxiliary; the broader archive
remains a data priority and any new training needs its own fixed protocol.

Two exact source/target binding tests and three text-overlap tests passed under
locked Pixi. The join and full text guard both exited0. The independent join-to-
guard check replayed all4,949 label/source bindings. No test labels, semantic
judgments or source files were regenerated or replaced.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_cached_stored_summary_join_v1`: complete candidate pairs and matching
  feature/source metadata, with input and output hashes.
- `20261025_shared_summary_open_source_guard_v1`: all66,074 retained metadata
  records and the empty conflict file; complete exclusion accounting.
- `20261025_cached_stored_summary_guard_binding_v1`: independent4,949-pair binding
  to the completed text guard.

The ongoing generated-code decoder study still uses its original732 engineering
TRAIN pairs. Its semantic result and the95%task objective remain separate from
these data-preparation findings.
