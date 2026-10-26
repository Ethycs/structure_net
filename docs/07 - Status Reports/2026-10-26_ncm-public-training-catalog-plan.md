# Full public paired-data catalog with protected reconstruction controls

Study `ncm-public-training-catalog-v1` prepares every eligible pair from the
unchanged pinned XSum TRAIN shard for a subsequent conditional pretraining
study. It does not select a smaller subset to fit an eager feature cache.
The previous all-row inspection found163,051 candidates among204,045 rows.
This phase admits a data catalog only after independent replay; it does not
launch model training, certify reference factuality or measure task accuracy.

Keep all320 public codec diagnostic articles as PUBLIC_DEVELOPMENT controls:
256 representative,32 long stress and32 numerical stress. Their IDs and
unchanged references were fixed before codec outcomes. Exclude from new TRAIN:
their article IDs, exact source hashes, exact reference hashes, and equivalent
native128 target frames. Apply the existing normalized-text/5-gram source
overlap rule to these320 articles as well: normalized equality, or at least32
common distinct word5-grams covering at least80% of the shorter set. This
reservation is prospective for conditional training; the earlier target-aware
codec diagnostic remains an exploratory component result.

Retain every exclusion from the prior inspected population, which checked
192 open application sources,512 protected components' metadata, and1376
current TRAIN sources. Never open or hash sealed final source files. Also
exclude exact/native-equivalent targets of the64 reserved application codec
controls and1376 existing reviewed TRAIN summaries. Deduplicate the remaining
public targets by native frame, retaining the first eligible original row.
Public article IDs identify source units; this is not a claim to reconstruct
all news-story families or detect arbitrary semantic paraphrases.

Recheck complete source and target hashes, original row identities, native
tokenization and application token bounds on every candidate. Preserve
original strings. Sources must remain192–2048 cl100k tokens; references must
be1–128 cl100k tokens and1–126 native lexical tokens, without special/unknown
tokens. Frame them as `[101, lexical..., 101, PAD...]` at128 positions. No
truncation, reference rewriting, outcome-based selection or teacher requests.

Write an indexed UTF-8 JSONL training catalog with byte offsets for bounded
random access, separate public development cases, explicit source/target/native
frame protection metadata, exclusion records and exhaustive counts. Source-only
projection must remove all target fields before body/tokenizer preparation.
Keep original dataset revision, article ID and row number with every pair.

Run CPU contract tests before the full prepare pass. Independently replay all
original source/target bindings, target frames, index offsets, exclusions and
new public-development text overlap after preparation. Admit the resulting
catalog for public-reference pretraining only if the replay succeeds. A
reference's presence in XSum does not establish factual correctness. Final
application qualification continues to use the unchanged complete-summary
semantic rubric and source-disjoint protocol.

Fresh roots under `data/experiments/ncm_paper_regime`:
`20261026_public_training_catalog_v1` and
`20261026_public_training_catalog_audit_v1`. Use locked Pixi `ncm`, CPU only,
from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_public_training_catalog prepare
```

The subsequent feature pipeline must retain full original-body FP32 features
and the pinned original prompt/tokenization. Use a bounded disk buffer or
on-demand extraction rather than materializing the estimated1.1TB feature
corpus. Its scheduling, memory, replay and optimizer lifecycle are separate
qualifications. No training budget or architecture change is selected here.
