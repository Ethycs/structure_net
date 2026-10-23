# Diagnostic pilot cache correction

Date: 2026-10-22, before the 64-case forward diagnostic. The first two-case
diagnostic pilot completed, but its native checkpoint failed saved-output
agreement on both cases. The new diagnostic wrapped both arms in one BF16
autocast scope and loaded different weights into the same parameter storage;
cached BF16 copies can therefore outlive the checkpoint they represent.

Invalidate that pilot for scientific interpretation. Preserve all its artifacts
under `20261022_diagnosis_pilot_v1` and add an invalidity receipt. Correct the
diagnostic to clear the autocast parameter cache after each checkpoint load,
snapshot the corrected source, and use a new pilot root
`20261022_diagnosis_pilot_v2`. The unused full diagnostic root is unchanged.
No training or new task grading is introduced, and no quality-based selection
of cases, checkpoints or decoding parameters is permitted.

The original training/evaluation code uses separate autocast scopes per update
and inference call, with checkpoint loads outside those scopes; this defect is
in the new diagnostic, not evidence of that defect in the completed training.
The corrected pilot must reproduce saved native and hybrid argmax output IDs
on both cases before launching the full forward-only diagnostic.
