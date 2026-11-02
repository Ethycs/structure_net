# Exhaustive replay of public source metadata

This systems audit implements the independent audit required by the source
metadata plan. It does not change the producer, selected rows, tokenizers,
prompt, alignment, training admission or quality endpoint. The full producer
is still running when this audit is specified.

After an actual successful producer exit, replay every saved record against
its original admitted pair. Verify contiguous byte offsets, per-record hashes,
source identities and strict source-only fields. Recompute all native prompt
IDs, GPT2 source IDs and character-overlap weights using the frozen original
preparation function, invoked separately from the public producer wrapper.
Require exact equality of the entire prepared record. This is an independent
execution and binding audit, not an independent implementation of tokenization.
Check nonempty normalized finite alignment weights and valid prompt positions.
Reproduce the original two TRAIN geometry controls and all nine public pilot
records. Reconcile every count and maximum against the producer result.

First audit the nine-row pilot using one CPU worker. Only after its actual
successful exit, audit all 162,353 full records using four spawned CPU workers,
128 records per task and at most eight tasks pending. Workers read indexed
records from disk; they do not hold the whole corpus or feature tensors. Save
counts and identities of the longest source and native prompt for a later
feature extraction pilot. All model weights and GPUs remain unused.

Use fresh roots `20261026_public_source_metadata_pilot_audit_v1` and
`20261026_public_source_metadata_full_audit_v1` under the existing NCM data
directory. Hash the audit source, tests, plan and all producer inputs/outputs
before execution and verify them again before admitting the result. A failure
is preserved and admits no metadata. No quality, factuality or speed claim is
made from this audit. Run through locked Pixi `ncm`, CPU only:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.audit_ncm_public_source_metadata pilot
```

Then use `full`, after recording both pilot-audit and full-producer exits.
