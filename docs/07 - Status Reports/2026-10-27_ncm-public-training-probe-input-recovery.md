# TRAIN probe: prepared-input identity correction

The v2 pilot exited 1 before producer creation or model loading (session
60840, chunk b6715a). It incorrectly compared prepared cases containing oracle
codes against `broad.CASES_SHA`, which identifies the original catalog.
The original BROAD training producer independently binds both files:

- catalog: `cd3d2a8a1fe0d398d1ca951e0df382d05c47cefe4d07892107b23d6e1c7e5914`
- prepared cases: `c04a84a4e691edb2237dac815b7b962a8dd4d9773fca77bf487334e5ef83c0ad`

Both current files match those prior bindings. Preserve v2 source commit
92aa082, qualification, and failure under
`20261027_public_epoch2_TRAIN_probe_launch_failure_v2`.

Version 3 verifies every consumed preparation file and source manifest against
the already authenticated original training producer. Apply the catalog hash
only to the catalog. Keep the complete 1,376-case population and original
16-control selection. Run focused contracts and the actual prepared-input
check on CPU before launching a fresh v3 pilot.

Use schema `nal.ncm-public-training-probe.v3` and fresh
`20261027_public_epoch2_TRAIN_probe_{pilot,run,audit,results}_v3` roots.
All scientific settings, comparisons, tolerances, memory gates, and the
unchanged epoch4 training service remain as originally specified. There was
no model inference or quality outcome in either failed initialization.
