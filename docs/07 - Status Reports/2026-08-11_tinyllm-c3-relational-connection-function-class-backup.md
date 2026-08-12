# TinyLLM C3 relational connection function-class backup receipt

**Status:** VERIFIED FINAL BACKUP

**Date:** 2026-08-11

## DVC

```text
root:   0dae52419be2a9e62f04cc138e36909f.dir
size:   54,465,356,747 logical bytes
files:  4,160
```

The push uploaded five new objects. An immediate replay reported
`Everything is up to date`, and local `dvc status` reported the data graph up
to date.

Primary and meta-hypothesis artifacts:

```text
data/experiments/tinyllm_c3_relational_connection_function_class/
  20260811_preregistered/result.json
SHA-256: 2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c

data/meta_hypotheses/tinyllm-c3-relational-connection-function-class-v1.json
SHA-256: 495cd6fdf3e2581082faf9fc37d19fa30eb5d49adc689c24a533150cf385a473
```

## lakeFS

```text
branch:  lakefs://artifacts/main
commit:  8eb97724e4bddf5e1bef57a1d7179b5c2d4e769d41aaac533fa64d8f69679d38
parent:  711602369a9ad3331ae22d1954742951aee6463571baa8485d1c13ba40f0cf9b
message: Seal TinyLLM C3 connection function-class evidence
```

The post-commit branch diff is empty. Direct object inspection recovered
checksum `0dae52419be2a9e62f04cc138e36909f` for the `668,323`-byte DVC
directory manifest.

## Verification

The expanded focused C3 suite completed as:

```text
276 passed, 1,483 deselected, 18 warnings
```

The meta-hypothesis store read back the function-class hypothesis and its
single no-training experiment record successfully.
