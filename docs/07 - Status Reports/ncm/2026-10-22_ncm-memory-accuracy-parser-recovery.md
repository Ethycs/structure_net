# Accuracy grading: conservative-failure parser recovery

2026-10-22. Before inspecting any model task-pass aggregate, the primary run
produced seven protocol errors and one terminal invalid packet among the first
16 completed packets. The error was the validator equating `faithful=false`
with an explicitly unsupported extractable claim. An unreadable candidate can
contain a supported extractable fragment and additional unparseable nonsense;
the judge can conservatively mark the whole candidate unfaithful. That is a
valid failure, not grounds for asking for another judgment.

Accept a false faithfulness verdict without requiring a negative extracted
claim. Continue rejecting `faithful=true` unless there are nonempty supported
claims with valid citations. Do not change any judge boolean, rubric, candidate,
pass rule, threshold or source evidence. This correction cannot turn any false
faithfulness verdict into a passing summary.

Stop the two active workers, preserve the complete V1 tree and interruption
receipt, and continue in `20261022_accuracy_v2`. Import the **first** reply per
packet that is valid under the corrected validator, including previously
rejected conservative failures. Preserve original responses and validation
metadata with origin hashes. Ignore later redundant replies after that first
valid judgment. Request only packets without a usable saved reply. Qualification
controls are rechecked from their saved replies under the corrected parser.

Primary qualification had already accepted 24/24 positives and rejected 24/24
negatives. Secondary qualification had nine completed valid packets when
interrupted. No primary task-pass distribution was inspected. V2 remains
exploratory; this is parser recovery, not rubric or outcome retuning.

To reduce grading latency, primary execution uses eight workers after an
eight-request systems concurrency pilot on controls. If that pilot fails,
retain the established four-worker primary execution. Secondary qualification
remains serial. The independent 64-case audit uses four
request workers after a four-request systems concurrency pilot on controls,
excluded from accuracy evidence. This changes throughput only: prompts,
candidates, model, temperature and decision rule are unchanged. Preserve its
pilot responses and require all four transport/schema outcomes valid before
the parallel secondary audit. If the secondary concurrency gate fails, retain
its failure and report that the secondary audit did not run.
