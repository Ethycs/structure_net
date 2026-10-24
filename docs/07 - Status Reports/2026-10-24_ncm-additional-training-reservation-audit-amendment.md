# Metadata reservation: audit serialization amendment

2026-10-24. The producing source684ad6d and both completed measurement roots
remain immutable. Actual full workflow47878 exited0 (chunk72c7d5); original CPU
finalizer33480 exited1 (chunk963a1e) at the deterministic availability comparison.
Its failed tool receipt is preserved in the fresh audit_attempt_v1 root.

The producer saved integer band keys through JSON, which necessarily reads them
as strings. The auditor compared that saved nested dictionary directly with the
in-memory integer-keyed expected dictionary. A fresh v2 auditor applies a JSON
roundtrip ONLY to the expected availability value at that serialization boundary.
It retains exact selected-row comparison, all10000 quotas, all component exclusions,
metadata membership, original source/input hashes and actual terminal checks.

Freeze the new auditor before use; write only fresh results_v2/report-v2 paths.
Retain the first failure as an artifact. No source files, measured selections,
semantic judgments, model weights or task-quality criteria change. Register only
after the fresh independent CPU audit actually passes. No quality or speed claim.
