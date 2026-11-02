# Exact deployment input-diversity correction

Date: 2026-10-10, after inspecting the first full run's timing outcomes.
Hypothesis `llama32-3b-lossless-deployment-efficiency-v1`, EXPLORATORY.

The independent finalizer rejected `20261010_v1/deployment/result.json` because
the two nominal benchmark sequences had identical first 128 tokens. The runner
checked inequality of their full 512-token sequences but did not check each
truncated control. Thus the 128-token prefill and generation correctness controls
were duplicates. Their computed outputs did match bitwise and their timings
remain diagnostics, but that run does not pass the declared diverse-input gate
and was not registered as a successful deployment result.

Known rejected-run timing ratios were 1.448× (128 prefill), 1.012× (512 prefill)
and 2.551× (128 prompt plus 32 generated tokens). Do not pool them with corrected
measurements or conceal the audit failure.

Fix the selection rule before rerunning: the first correctness input is the
first `length` tokens of benchmark sequence 0; the second is the final `length`
tokens of benchmark sequence 1. Assert inequality for each workload before
capture or measurement. The timed input remains the unchanged first sequence.
All precision, weights, arithmetic, timing samples/order, output ownership,
memory accounting, thresholds and hardware controls remain unchanged.

Add a regression test with two sequences sharing a prefix but differing in
their tails. Run corrected tiny/full lifecycle controls in
`20261010_pilot_v2`, then all three full workloads in `20261010_v2`, under
`data/experiments/llama_lossless_deployment/`. Preserve the earlier artifacts,
record this amendment in the new producer fingerprints, and use only corrected
full-run timings as primary evidence. The finalizer must reject duplicate
correctness inputs again if this requirement is ever violated.
