# TinyLLM C3 relational-connection acquisition backup receipt

**Status:** VERIFIED FINAL BACKUP

**Date:** 2026-08-11

## DVC

```text
root:   a1a34ac19f4b6b91ad02f779d6968922.dir
size:   54,480,674,615 logical bytes
files:  4,218
```

The push uploaded 60 new objects. An immediate replay reported
`Everything is up to date`, and local `dvc status` reported the data graph up
to date.

Primary, corrective, and meta-hypothesis artifacts:

```text
data/experiments/tinyllm_c3_relational_connection_acquisition/
  20260811_preregistered/campaign_results.json
SHA-256: b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a

data/experiments/tinyllm_c3_relational_connection_readout_audit/
  20260811_artifact_audit/result.json
SHA-256: 1fb139ebd13b1ac78d77fa0b82d206f237b3d14ef86173cd7e8d6825dd1731a5

data/meta_hypotheses/tinyllm-c3-relational-connection-acquisition-v1.json
SHA-256: ef16068b31147466a125aa0ba9eacc59e5b4d507fd87bce837099bf22c9f829e
```

The preregistered primary classification is
`exact_function_class_but_population_acquisition_unreliable`: the analytic
positive control passed in five of five seeds, while the learned true
connection arm passed in one of five and every information control passed in
zero of five. The transparently post-outcome, artifact-only readout audit did
not alter that primary result. It showed that three additional seeds had
learned the relation up to public scalar calibration, yielding four of five
under a sealed affine or neutral-carrier readout, while one seed retained the
wrong winding.

## lakeFS

```text
branch:  lakefs://artifacts/main
commit:  3ca21a4c6119bb831202e9857d37b7cb0153790c60213d30d31a7f0180cb77a5
parent:  8eb97724e4bddf5e1bef57a1d7179b5c2d4e769d41aaac533fa64d8f69679d38
message: Seal TinyLLM C3 relational connection acquisition evidence
```

The post-commit branch diff is empty. Direct object inspection recovered
checksum `a1a34ac19f4b6b91ad02f779d6968922` for the `678,701`-byte DVC
directory manifest.

## Verification

The broad TinyLLM and meta-hypothesis regression completed as:

```text
1,445 passed, 1 skipped, 327 deselected, 18 warnings
```

All 20 learned campaign cells passed the declared action, reload, resume, and
CUDA lifecycle checks. The campaign used 48,000 primary optimization steps and
24,000 resume-verification steps; no TinyLLM training was performed.
