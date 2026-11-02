# NCM memory training: lifecycle parser fix

2026-10-22, before any optimizer update or test output. The first CUDA pilot
loaded the original models but failed while reading JSONL: Python's
`str.splitlines()` treats Unicode paragraph separators inside JSON strings as
record delimiters. Read physical file lines instead. A Unicode regression test
checks the actual failure. Data and all training budgets/objectives are unchanged.

Keep the failed systems-only root `20261022_pilot_v1` and its source snapshot.
Run the corrected pilot in `20261022_pilot_v2`, then the originally planned
training root `20261022_training_v1`. This is an implementation recovery before
training, not a retry of a quality outcome. The original plan remains unchanged.
