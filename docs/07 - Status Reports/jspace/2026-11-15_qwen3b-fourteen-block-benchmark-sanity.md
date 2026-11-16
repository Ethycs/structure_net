# Fourteen-block Qwen benchmark sanity check

Status: prospective exploratory check, specified before new benchmark outputs.
Study ID: `jspace-qwen3b-fourteen-block-sanity-v1`.

The question is whether the current compressed model retains the original
model's multiple-choice competence on examples outside recovery training and
development. Recovery and further depth expansion remain stopped during this
comparison. This is a sanity check, not the final all-block confirmation.

The candidate is the accepted cut22 checkpoint from
`jspace_3b_feature_resident_campaign/20261115_v1/stages/block_22/verify/accepted.pt`,
SHA256 `814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401`.
It contains compressed blocks22–35 and the trained final norm; native blocks0–21
and the original tied embedding/readout are retained. It was selected at joint
epoch1 and freshly verified at50/64 on reused development. Its native reference
is the same pinned official Qwen2.5-3B checkpoint used by the parent campaign.

Freeze128 new questions,32 each from ARC-Challenge, BoolQ, HellaSwag and
WinoGrande, using the pinned local evaluation-source Parquets, seed582463,
the original prompt/tokenizer and384-token input cap. Exclude all historical
training/development/evaluation identities, the unused recovery expansion and
the reserved512 final examples. Authenticate the reserved IDs using the existing
identity-only manifest; do not open the final questions or labels. Preserve
source-row provenance, tokenization and file hashes in a new data root.

Score both arms in BF16, eager attention, singleton unpadded prompts, TF32 off,
and identical single-letter choice-row readouts. The native arm may also capture
its exact pre-layer22 activation for compact-suffix evaluation; this is the
unchanged native prefix of the hybrid model, not a teacher-provided answer.
Validate the boundary against existing native capture controls and strictly
reconstruct/pack the selected suffix and norm. The training development score
used batch4/full-vocabulary readouts, so its50/64 is not a numerical equality
target for the new singleton choice-row scoring protocol.

Report total and per-family counts, paired gains/losses, prediction agreement
and the exact paired sign/McNemar test. The prespecified descriptive screen is
at most6 net fewer correct answers than native on128 (a drop below5 percentage
points). This small sample cannot establish a population5pp noninferiority
bound. No significant difference is not proof of equivalence. Preserve all
family regressions even if the aggregate screen passes. No training, checkpoint
selection, repeated sampling or altered scoring after seeing these outcomes.

Use Pixi and the experiment reservation controller for sequential GPU phases
on the3060. Freeze the outcome-relevant producer, candidate and dataset before
scoring. Abort on failed provenance, boundary, pairing, finiteness or strict-load
controls; retain failed artifacts. Do not infer generation quality, speed,
full36 completion or a broad model leaderboard score from this restricted test.

Planned data root: `data/experiments/jspace_3b_sanity_data/20261115_v1`.
Planned evaluation root: `data/experiments/jspace_3b_sanity_evaluate/20261115_v1`.
