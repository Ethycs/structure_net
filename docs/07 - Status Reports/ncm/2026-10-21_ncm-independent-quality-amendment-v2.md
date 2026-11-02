# Judge transport amendment, before primary generation

Date: 2026-10-21. Hypothesis: `llama32-3b-ncm-independent-quality-v1`.
This amends the [original preregistration](2026-10-21_ncm-independent-quality-preregistration.md).
All data, rubric, checkpoint, endpoint, target and selection rules remain fixed.

The GPT gateway's four-worker calibration returned repeated HTTP500 errors,
including pairs exhausting all three attempts. Its required complete calibration
became impossible; the unfinished calibration was stopped and retained as invalid
infrastructure evidence. No primary or stress-model outputs have been generated
or inspected. Both excluded two-case CUDA systems pilots completed.

Use the separately available `claude_code/claude-sonnet-4-6` as primary judge,
still requiring the same 48-summary known-answer gate. Attempt
`codex_sdk/gpt-5.6-sol` as the secondary judge with **one** network worker, to
test whether avoiding concurrent requests fixes its transport. Primary uses at
most four network workers. Never run both judge jobs simultaneously. Stop a
calibration once an exhausted infrastructure error makes its complete-data gate
impossible; record the stopped calibration, with no accuracy interpretation.
If the secondary cannot calibrate, report agreement as unavailable, while keeping
the calibrated primary endpoint. This follows the original secondary-failure rule.

Use fresh roots under
`data/experiments/ncm_quality/20261021_judge_v2/`; preserve v1 roots.
Reuse the prospectively frozen `20261021_data_v1` sources and calibration cases;
do not resample them. Rerun the excluded CUDA pilots for the current producer.
Completed prior producers are verified against their archived source snapshots
and unchanged inputs; current workers still check live source/input hashes at
completion. A new dependency manifest precedes primary generation. Freeze both
this amendment and the original plan. No model-quality-driven changes are allowed.
