# Available source–summary pair census

The user redirected the work from another architecture experiment to establishing the total data supply. The prepared 1,376/2,384 application pairs are not an inventory of everything available. No new architecture training is launched during this census; the two already-running trainers continue unchanged.

The published handoff contains 798 archive bundles, with 370 containing files selected by its summary-artifact index, plus the separately published 500 core stores. Restore all indexed summary payloads from immutable lakeFS commit `ded03c25a12e56d620710d2bff334ba5beed923584783c460084eb5922a8270a`. Verify DVC MD5 and archive SHA256, and selected member SHA256. Preserve metadata, receipts and failures. The index covers known summary-related filename classes; it does not prove that every other archived file contains no usable data.

Count physical records, distinct exact input–target pairs, and distinct exact inputs separately. Global pair deduplication must ignore source namespace aliases. Several different targets for one input count as several pair records but one input. Do not equate those counts with independent source groups.

For stored atomic sections, reconstruct every source span from its source turn and check bounds, whole-turn hash, span hash and attribution. Separate exact copies from summary candidates. For hierarchy records, reconstruct the ordered child-summary input where all children are available; flag unavailable raw-evidence provenance separately. Structural records without complete child inputs remain a separate unresolved category, rather than silently disappearing from the denominator.

Audit atomic and merge caches and saved model request/response receipts. A cached target with only a source hash is not yet a recovered pair. A receipt containing only a prompt hash is not an exact prompt. Preserve raw-batch and merge distinctions, failed outputs, and records with missing inputs. Request-level batch pairs and fragment-level pairs must not be added as though they were independent populations.

Report application input/output length coverage using `cl100k_base` and distinguish exact copies, short bypasses, long inputs, and targets over 128 tokens. These are structural and length counts, not human-reviewed labels, semantic accuracy, or training admission. Source-group split checks remain necessary before any enlarged corpus is trained. No final evaluation file or judge result is used by the census.

The census uses locked Pixi environments and CPU-only, disk-backed tables. Archive payloads reside on `/data2`. Initial random SQLite I/O on that rotating disk was slow; the scanner was deliberately stopped after 24 committed stores, and a SQLite backup preserved those records on SSD. The original scanner is not reported as a completed run. The unchanged scanner continues under a separate SSD runtime record. Two focused counting tests pass with the repository-level unrelated conftest excluded; the initial test command failed to import an unrelated SciPy dependency.

Artifacts: `data/experiments/ncm_paper_regime/20261027_available_pair_census_v1` and `20261027_available_pair_census_fast_v1`. Payload storage: `/data2/structure_net/ncm_available_pairs_20261027_v1`. The current restoration and census are running; a completed total is not yet claimed.
