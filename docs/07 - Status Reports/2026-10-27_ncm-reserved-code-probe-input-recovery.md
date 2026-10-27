# Reserved-code probe: correct the inspection audit input schema

Version1 pilot stopped before model construction or prediction. Actual unified
exec session10054 ended with exit1, chunk c27058: its input list named
`20261026_public_xsum_train_inspection_audit_v1/campaign_results.json`, which
does not exist. That historical audit publishes `result.json` and its actual
receipt; both were already in the input list and were successfully verified by
the CPU source guard. The source qualification did not check completeness of
the later fingerprint input list. Preserve the failed pilot, source snapshots
and actual failure receipt unchanged.

Version2 removes only that nonexistent input, retains the actual audit result
and receipt, and adds this amendment, its tests and the failed-run evidence.
Require every input path to exist before the new pilot creates its output root.
No cases, checkpoint, training or scoring code, thresholds, GPU, precision or
pilot repetitions change. New roots end in `_v2`; version1 is never reused.
Run the original ten CPU tests and recovery-contract tests before retrying.
The complete scientific protocol remains the original reserved-code probe plan.
