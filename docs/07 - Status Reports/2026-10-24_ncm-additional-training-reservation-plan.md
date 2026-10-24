# Additional training-source availability and metadata reservation

2026-10-24. Hypothesis `ncm-additional10000-training-source-reservation-v1`.
This CPU control runs while the immutable1024 word-gradient study trains.
It does not change that study or generate any model outputs or judgments.

Use only the previously registered complete source-component index and448-case
selection sidecar. Require parent campaign96f602f2..., original actual worker0,
exact metadata hashes and registered readback. Do not open, hash or reconstruct
any selected final source file. The public source corpus is not needed here.
Exclude all blocked components, which include all2828 known sources and320 old
reserved identities, plus ALL448 newly reserved DEV/final components.

Run a two-per-length-bin10-case pilot and then reserve10000distinct additional
TRAIN components:5000 in192-511cl100k,1250 each in512-767,768-1023,1024-1535 and
1536-2048. Reuse the original deterministic augmenting-path quota matcher with
this new fixed namespace. Preserve each selected row's original metadata and
public provenance; no source text, target text, quality filtering or relabeling.
Abort rather than reduce any quota. Keep hostRSS<=8GB. No GPU work.

Freeze source/tests/plan before launch. Independent CPU audit must replay full
metadata admission and selected quotas, disjoint components, hashes and actual
worker/whole-tool exits. Register the metadata-only result with confirmedFalse,
task_accuracyNone and no quality/speed claim. Canonical semantic cache unchanged.
This establishes whether more training sources exist; labels, features and a
larger-data training study would require separate prospective artifacts. The
95%complete-summary goal and existing evaluation gates remain unchanged.
