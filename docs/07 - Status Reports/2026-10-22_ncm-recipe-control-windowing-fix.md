# Original-domain control reference windowing correction

Date: 2026-10-22. The original two-case CUDA pilot passed, and the unmodified
released shard generated all fixed 128 control cases successfully. The local
codec scorer then rejected a reference longer than 48 tokens. The authors'
reference summaries are not all bounded to one codec canvas; their VQ tokenizer
encodes long texts in consecutive windows.

Preserve the failed scorer root `20261022_control_v1` and its official generated
outputs. Correct only codec reference scoring: cover every token in consecutive
48-position windows and PAD only the final short window. No truncation, changes
to generation, sample selection, ROUGE, thresholds or model weights. Add a
windowing unit test, a corrected pilot `20261022_control_pilot_v2`, and a new
control root `20261022_control_v2`. Reuse the original completed shards after
checking cohort, input hashes and unchanged vendor generation source; retain
reuse SHA receipts. The original 128-case scorer produced no terminal verdict.
No new training runs until the corrected control gates pass.
