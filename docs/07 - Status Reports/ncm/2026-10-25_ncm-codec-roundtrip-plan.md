# Direct summary-codec capability diagnostic

Exploratory hypothesis `ncm-summary-codec-roundtrip-v1`. Written before these
outputs are generated. No training, checkpoint selection, semantic judge calls,
speed benchmark, or sealed final-test access is part of this diagnostic.

Question: can the existing codec reconstruct complete summaries when given
codes encoded from those same summaries? Compare the released LM1B codec, the
repaired parent codec, and the latest decoder adapted to generated codes. For
the two application checkpoints, also retain their existing source attention
and provide the correctly bound frozen source features. The codec-only arms
receive no source information. The generator is bypassed in every arm.

Use all 64 already reserved Sol teacher/component summaries, without selecting
by individual judgment outcome, plus the existing fixed 16 engineering TRAIN
diagnostics. Check identity disjointness against the 732 engineering examples
and the earlier native training population. Reuse the original summary texts
and full native CLS framing; never truncate. Report tokenizer normalization,
unknown tokens, and frame overflow separately. These 64 cases are a development
component control, not an untouched final test or evidence of source-only
summarization accuracy. Teacher label correctness is not tested by reconstruction.

Run CPU preparation and focused metric/selection tests first. Then run a CUDA
pilot on the first and longest reserved summary, requiring repeated identical
codes and outputs. Run the five fixed arms on all 80 cases, one case at a time,
in FP32 with TF32 disabled on physical GPU1 (RTX 2060), using locked Pixi `ncm`.
The existing lossless disk feature loader has a 256 MiB cache. Preserve any
systems failure and use a new artifact root for scientific changes.

Primary measurements, separately for the two populations and all five arms:
exact decoded text versus the native-tokenizer round trip; exact active token
sequence including boundaries; content-token agreement excluding PAD/CLS;
complete preservation of the reference's digit sequence among cases with digits.
Digit-sequence equality is a mechanical diagnostic, not numerical reasoning or
unit fidelity. Exact text/sequence results also cover names and units; inspect
mismatched examples rather than inventing an entity-accuracy score.

A descriptive reconstruction gate requires at least 61/64 exact native-rendered
summaries and at least 99% content-token agreement on the reserved control.
This gate is deliberately stricter than semantic equivalence and cannot be
interpreted as complete-summary task accuracy. Failure does not prove a code
capacity impossibility; decoder changes and distribution mismatch remain possible.

Artifacts: `data/experiments/ncm_paper_regime/20261025_codec_roundtrip_{prepare,pilot,eval}_v1`.
Record model/input/source hashes, all per-case codes and outputs, GPU peak memory,
the actual process receipts, and a replayed aggregate. Earlier completed evidence
and reports remain unchanged.
