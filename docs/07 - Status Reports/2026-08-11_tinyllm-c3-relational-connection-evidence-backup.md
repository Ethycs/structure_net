# TinyLLM C3 relational-connection evidence backup receipt

**Status:** VERIFIED FINAL BACKUP

**Date:** 2026-08-11

## DVC

```text
root:   b0f87c2fff6e69a7d1b18dff01a6a547.dir
size:   54,465,278,743 logical bytes
files:  4,157
```

The push uploaded nine new objects. An immediate replay reported
`Everything is up to date`, and local `dvc status` reported the data graph up
to date.

Primary and meta-hypothesis artifacts:

```text
data/experiments/tinyllm_c3_relational_connection_preflight/
  20260811_preregistered/result.json
SHA-256: ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e

data/meta_hypotheses/tinyllm-c3-relational-connection-preflight-v1.json
SHA-256: 967413af162fb05a4539ec5ffc31c92708a23f6d6d5a5fff4c0b79ae682a0522
```

## lakeFS

```text
branch:  lakefs://artifacts/main
commit:  711602369a9ad3331ae22d1954742951aee6463571baa8485d1c13ba40f0cf9b
parent:  6defbdd07681ff5aa6de70438619a9a4b5e6f49bb7fb3a45c20a37cc0c6c5999
message: Seal TinyLLM C3 relational connection preflight evidence
```

The post-commit branch diff is empty. Direct object inspection recovered
checksum `b0f87c2fff6e69a7d1b18dff01a6a547` for the `667,940`-byte DVC
directory manifest.

## Verification

The expanded focused C3 suite completed as:

```text
263 passed, 1,483 deselected, 18 warnings
```

The meta-hypothesis store read back the hypothesis and all five seed-level
experiment records successfully.
