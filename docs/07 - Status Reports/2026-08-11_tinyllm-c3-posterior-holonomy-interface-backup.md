# TinyLLM C3 posterior-holonomy interface backup receipt

**Status:** VERIFIED FINAL BACKUP

**Date:** 2026-08-11

## DVC

```text
root:   125447de251dda33ab40e06d680cbb17.dir
size:   54,480,832,815 logical bytes
files:  4,232
```

The push uploaded nine new objects. An immediate replay reported
`Everything is up to date`, and local `dvc status` reported the data graph up
to date.

Primary and meta-hypothesis artifacts:

```text
data/experiments/tinyllm_c3_posterior_holonomy_interface/
  20260811_preregistered/result.json
SHA-256: 102125d3c465a30be64a51b6a3b3a59ebb8c350dfb92a562f72684431b4601fc

data/meta_hypotheses/tinyllm-c3-posterior-holonomy-interface-v1.json
SHA-256: 7023a27783f96d3d1a20a12019ff98dd0225c32e8436e37196062e1aa7a28e8b
```

The locked classification is
`posterior_holonomy_moment_exact_soft_interface`. All `2,080` posterior
simplex points, `4,259,840` posterior/phase cells, three coordinate shifts,
and ten frozen module/shift replays passed. The maximum frozen replay error was
`2.38e-7`; no checkpoint state changed, no fitting occurred, and no TinyLLM
model was instantiated.

## lakeFS

```text
branch:  lakefs://artifacts/main
commit:  ddc21b1a024b1fa6fb49bde3cca7d1e60bd4f0eff6a86e79d7c32c83620e2fe4
parent:  7e5ccb31dbc56426595754064e26238819f9ebf4216e6ed5ceaee9df988dfd85
message: Seal C3 posterior holonomy interface evidence
```

The post-commit branch diff is empty. Direct object inspection recovered
checksum `125447de251dda33ab40e06d680cbb17` for the `680,239`-byte DVC
directory manifest.

## Verification

The posterior-interface plus predecessor focused suite completed as:

```text
27 passed, 18 warnings
```

The broader TinyLLM/meta-hypothesis regression then completed as:

```text
1,472 passed, 1 skipped, 327 deselected, 18 warnings
```

The meta-hypothesis store read back the confirmed hypothesis and all five
frozen source-seed experiment records successfully.
