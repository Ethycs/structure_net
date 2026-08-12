# TinyLLM C3 connection-observation identifiability backup receipt

**Status:** VERIFIED FINAL BACKUP

**Date:** 2026-08-11

## DVC

```text
root:   17f42a58ddb17368d1238fc66b1da65e.dir
size:   54,480,788,023 logical bytes
files:  4,225
```

The push uploaded nine new objects. An immediate replay reported
`Everything is up to date`, and local `dvc status` reported the data graph up
to date.

Primary and meta-hypothesis artifacts:

```text
data/experiments/tinyllm_c3_connection_observation_identifiability/
  20260811_preregistered/result.json
SHA-256: 23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c

data/meta_hypotheses/
  tinyllm-c3-connection-observation-identifiability-v1.json
SHA-256: 7909999fcac9d913470090f29cb917044d5510761d0804daeca6bd53bd70d155
```

The locked classification is
`total_holonomy_minimal_known_noise_analytic_no_training_scope`. Total-only
connection replacement changed no analytic or frozen learned prediction in ten
fresh cells; all seven exact erasure collisions passed; and all five exhaustive
known-noise enumerations matched the closed form below `1.2e-15`. The audit
performed zero optimizer steps and instantiated zero TinyLLM models.

## lakeFS

```text
branch:  lakefs://artifacts/main
commit:  7e5ccb31dbc56426595754064e26238819f9ebf4216e6ed5ceaee9df988dfd85
parent:  3ca21a4c6119bb831202e9857d37b7cb0153790c60213d30d31a7f0180cb77a5
message: Seal C3 connection observation identifiability evidence
```

The post-commit branch diff is empty. Direct object inspection recovered
checksum `17f42a58ddb17368d1238fc66b1da65e` for the `679,480`-byte DVC
directory manifest.

## Verification

The combined source-acquisition and observation-audit focused suite completed
as:

```text
30 passed, 18 warnings
```

The broader TinyLLM/meta-hypothesis regression then completed as:

```text
1,461 passed, 1 skipped, 327 deselected, 18 warnings
```

The meta-hypothesis store read back the confirmed hypothesis and all five
frozen source-seed experiment records successfully.
