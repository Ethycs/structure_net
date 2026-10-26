# NCM depth comparison: reporting and candidate selection

Prospective reporting supplement to the unchanged depth plan. Written after the
24-block arm completed 1,024 updates and before either new arm generated or
graded DEV summaries. The 12-block control has started. No new semantic outcome
has been inspected. This supplement changes no training configuration or gate.

Replay the existing 128 source-disjoint DEV outputs, exact judge packets and
canonical first judgments for each arm. Preserve the initial 4,179 first
judgments and all control-arm judgments in the deeper arm. Recompute semantic
pass counts and both 64-case cohort gates. Invalid judgments remain unresolved;
they do not become failures, successes or a reduced accuracy denominator.

Report each arm's valid judgments, complete-summary passes out of 128, cohort
counts, parameter count, training time and peak allocation. Report paired
decisions and the 24-minus-12 accuracy difference only with all pairs valid.
TRAIN word/code agreement is diagnostic only. Training time is not an inference
speed measurement. Initial architectures have different tensors and parameter
counts; describe the identity expansion and its four-case initial parity check,
not identical parameter initialization.

Only an arm passing the existing same-case Terra/Sol joint DEV gate in both
cohorts may become a candidate for the separately governed sealed final test.
If both qualify, choose higher joint DEV accuracy; break an exact tie in favor
of the 12-block model. If neither qualifies, select neither. DEV qualification
alone never establishes the final 95% task goal. Final320 stays sealed here.

Use independent per-arm replay artifacts and a matched aggregate, retain actual
tool exit receipts, then register the completed exploratory result and verify
exact, semantic and local-JSON registry readback. Preserve failed and invalid
outcomes. One training seed is not a replicated architecture result.
