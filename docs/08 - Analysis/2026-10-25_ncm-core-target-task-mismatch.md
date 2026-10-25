# Historical target matches are atoms, not complete-summary supervision

All 4,949 exact source matches passed raw source/target/frame and protected-family
replay, but the stored targets do not match the intended complete-summary task.
Examples include `That`, `Here`, and `Here is an example of how you can do this:`.
Other targets preserve a single extracted fact from a long, multifaceted source.
They can be useful atoms in the original memory system; the checks performed do
not qualify them as supervision for a whole source summary.

The diagnosis came from target distribution and raw inspection before any student
training. Of 4,949 targets, 1,164 have fewer than ten native lexical tokens, and
3,899 map to one root code. Root concentration alone is not evidence of a bad
dataset; the actual source/target task mismatch is the reason for exclusion.
The existing numeric/source eligibility result remains valid, and every label
and audit is preserved. No target was repaired or selectively rerolled.

The prospective historical error-generation worker was stopped after 100 per-case
predictions (96 in its last bulk checkpoint). Its exact owned PID/command was
verified before SIGTERM; the tool actually exited143. The unexecuted 5,681-pair
training proposal and tests were archived with hashes under
`data/experiments/ncm_paper_regime/20261025_core_target_task_mismatch_v1/`.
No student weight update used these labels. The 732 engineering TRAIN summaries
remain a separate candidate pool, with 156 additional engineering test inputs.
All original DEV/final/component exclusions stay intact.

The replacement protocol trains code correction on the engineering summaries
using actual frozen-generator errors and a word-level straight-through loss
through the frozen word decoder. All parent/MLP and original NCM weights remain
frozen. No accuracy or speedup is established by the data audit or this design.
