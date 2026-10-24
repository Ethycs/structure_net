# Reserve full size NCM evaluation sources with stronger split controls

Date: 2026-10-23. Prospective data preparation
`ncm-clean-full-size-source-partitions-v1`. The last dense student scored0/64
short and1/64 long under valid Terra judgments; the only passing output copied
a TRAIN target from an overlapping passage. The registered amendment found
four TRAIN/long overlaps and one pair within development. Existing final-sidecar
metadata also flags a related TRAIN family. No final source text was opened.

Prepare fresh128 development and320 final sources before any new model outputs
or labels on these cases. This replaces the primary source populations for
split validity while preserving the full task size, range and95% target. Keep
all original failed experiments and both original closed final source files
unchanged and unopened. This is data preparation, not a claim that a student
has reached the target or that the original paper was replicated.

Use only complete turns from the existing SHA-verified500-sample public
LongMemEval-S release, SHA256
d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442.
Use every public role and ignore question answers, relevance flags, old saved
summary targets, model outputs and quality scores. Candidate sources must have
192–2048 cl100k tokens. Larger public turns may connect duplicate families but
cannot be selected. Short bypass turns below192 tokens do not join unrelated
conversations through generic duplicate text. Never clip a selected source.

Exclude all690 approved TRAIN sources, the old128 development sources, the
original2048 training and64 development sources, and the old wider pool's112
training/16 diagnostic sources. Recover original256 final identity/turn metadata
from the existing public-proof sidecar and original raw native-test metadata;
use only identity/span fields. Include all64 old long-final sidecar identities.
Neither old sealed source file may be opened or hashed.

Group sources before splitting. Join known group aliases, observed conservative
source-family variants, identical public full turns, equal NFKC/case-folded word
sequences, and exact contiguous word5gram overlap: at least32 shared unique
5grams covering at least80% of the shorter set. Connected components propagate
every known-source exclusion through alias and overlap chains. Exact duplicate
text aliases share a node. Use one selected source per complete unblocked
component. Preserve an index of all components, source hashes and public origins
so future training can exclude every reserved component.

Reserve256 final short sources and64 development short sources in192–511 tokens.
Reserve64 final long and64 development long sources, each with16 sources per
512–767,768–1023,1024–1535 and1536–2048 band. Use deterministic augmenting-path
quota matching, with SHA256 namespace `ncm-clean-full-size-source-partitions-v1`
for component, row, cohort and public-origin ordering. Never shrink quotas on
failure. Select final/development membership after complete component allocation,
without inspecting outputs or labels. Verify literal complete public spans,
source-family/turn/text exclusion against all known sources, and pairwise
disjointness across all448 selected sources before sealing exports.

Run on CPU with CUDA explicitly hidden and a24GiB peak-RSS guard. First run
focused tests, including exhaustive small quota-feasibility comparisons and a
test that final stat verification cannot open file contents. A two-public-sample
lifecycle pilot builds the actual graph with all exclusion metadata, exports up
to one fresh source per available band, and requires exact public provenance and
JSON readback. Full preparation starts fresh with all500 public samples. Source
selection is not a model-quality measurement.

Write creation hashes and stat receipts for the two new final exports, then
seal them. Subsequent CPU finalization may replay component/identity/quota/span
metadata and check final file stats, but may not reopen or rehash sealed content.
It independently verifies development export bytes. Keep final creation hashes
outside the generic artifact-rehash list. Final content/provenance/text guards
must be independently checked when final evaluation becomes eligible; source
reservation alone cannot satisfy the model completion audit.

Freeze source, tests, plan and auditor before launch. A CPU supervisor owns
pilot, preparation, finalization and conservative registry/read-back stages with
actual child exits. Use fresh `20261023_clean_source_partitions_{workflow,pilot,results}_v1`
roots and `20261023_clean_source_partitions_v1` under the existing experiment root.
Report `clean_full_size_sources_reserved_quality_unmeasured`, or preserve any
systems/availability failure without changing the full quotas.

Future development still requires at least61/64 SAME cases passing both
qualified Terra and Sonnet in each cohort, all judgments valid. Final requires
244/256 short and61/64 long passing both, all valid, complete128-token summaries
graded for faithful/salient/readable content with paraphrases allowed. Both new
final cohorts remain closed until a fixed candidate clears development and
original frozen-body integration checks. No exclusions, repairs, clipping,
semantic rerolls or best-of generation. Preserve every canonical first judgment
and every previous failure. No inference-speed claim before verified quality.
