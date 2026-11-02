# Re-evaluate fixed native baselines under the qualified 128-token profile

Date: 2026-10-23. Exploratory `ncm-budget128-fixed-baselines-v1`.
Known old-profile results: unchanged 3B native parent 49/64, unchanged intact 1B
50/64. The earlier rubric included an approximately 32-word salience budget;
the application stores 128 tokens. Both aliases passed all 50 qualification
controls for the explicit budget128 amendment. Current NCM training is running;
its final checkpoint and development outputs have not been inspected.

Use only the already frozen 64 outputs per intact native baseline, from
`20261023_native_parent_control_v1` and `20261023_intact_decoder_outputs_v1`.
No model generation, text repair, clipping, case selection or prompt change.
Verify producer/input identities and output hashes. Require the exact ordered
original 64 development IDs, source IDs, groups and source hashes; verify public
source bytes. Preserve all old-profile judgments and aggregates unchanged.

The only intervention is the already qualified 128-token evaluation profile.
Use the exact source-grounded rubric, first-valid judgment cache, parser and
code-enforced cl100k storage cap already used for the current student study.
First-valid cache identity includes complete packet, rubric, role and model.
Reuse identical new-profile packets across conditions; do not reuse old-profile
decisions as new-profile decisions. Retain negative/invalid results and all raw
attempts. Maximum three transport/schema attempts, four API workers, no semantic
rerolls. Use only approved public data; no generation/judging API overlap.

A separate four-output lifecycle pilot uses the first two frozen outputs from
each baseline. Require all four valid first judgments, without a quality gate;
reuse those exact judgments in the full evaluation after actual terminal success.
Then grade every original case for 3B followed by 1B with qualified Terra.
For each baseline, require 64 valid and at least 61 complete-summary passes
before qualified Sonnet independently grades all 64. Sonnet uses the same gate.
Do not exclude over-cap, empty, unfaithful, unreadable or incomplete summaries.

Report counts, component judgments, storage overflows, invalids and Wilson
intervals beside the original old-profile counts. A failed baseline does not
stop the separately declared student training; a passing baseline cannot stand
in for student quality. No inference that rubric changes improve model weights.
The old 0/64 pruned readout failure remains a failure and is not regraded here.

Roots `20261023_budget128_baselines_{pilot,grading,results}_v1`. Freeze this plan,
implementation, qualification identities, original sources and frozen outputs.
Finalize conservative aggregate/report and register/read back after completion.
No NCM-success, compression, speedup, representative full-range or final-test
claim; the final 256 cases remain sealed.
