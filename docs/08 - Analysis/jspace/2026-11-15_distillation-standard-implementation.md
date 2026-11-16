# Distillation standard implementation evidence

The requested [distillation standard](../../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md)
and [implementation guide](../../02%20-%20Implementation/distillation-recovery.md)
are committed. Three parallel agents implemented response data, training, and
NAL lifecycle integration; the parent integrated the compiled-Qwen adapter.
Existing GPU experiment producers were not changed.

## Validation

- 53 focused CPU tests across response data, training, NAL lifecycle and Qwen
  integration pass under locked/offline Pixi `ncm` with CUDA hidden.
- Exact token-chunked loss values and gradients match the dense reference,
  including tied head/trunk parameters. Whole-response supervision includes
  the first response token and observed EOS. Padding, holes and wrong splits
  cannot silently become training targets.
- A tiny compiled Qwen trains and strictly reloads in FP32 and BF16 with both
  blocks using residual/head compression and polynomial MLP substitutions.
  Original-teacher state and fixed student geometry remain unchanged.
- Short-rollout tests score the exact student prefixes, exercise authenticated
  disk targets and a real recovery update, and reject changed provenance or
  excessive declared staleness. Capped outputs do not acquire an invented EOS.
- Persistent CPU lifecycle:
  `data/experiments/nal_distillation_standard/20261115_cpu_v2`.
  It performs two optimizer updates on 12 synthetic supervised tokens, saves
  real checkpoints, reloads into a separate model object, and reads the existing
  NAL result ledger. A second invocation resumes without retraining; a separate
  aggregate invocation writes `readback.json`.

The persistent classification is **`systems_passed`, `confirmed:false`**.
Its scientific fingerprint is
`d9f8f3046a5e521a552fd8eec3060c19fd9526e6405efd4c8afa26493f1bb9f8`;
producer fingerprint is
`275513fea74d141b1b33f959198f1ea3dcb928992ee7398faedee74392210e46`;
replicate result SHA-256 is
`d2a1f5bbfe47c17e78ec804e696c9063824e16f5335cd6e9b7aaba9cc2f8a2e6`.

The preceding `20261115_cpu_v1` attempt is preserved. Training, reload and worker
ledger admission succeeded, but the outer aggregation compared a relative
checkpoint path with the canonical absolute path. Commit `8e4723a` normalizes
that boundary and adds real relative-root and symlink-root regressions. The new
v2 attempt uses the corrected producer; v1 receipts were not overwritten.

## Corpus and remaining qualification

The [metadata inventory](2026-11-15_distillation-corpus-inventory.md) locates
208,631 prepared pairs and 259,636 overlapping archive candidates. These are
availability counts, not a newly admitted recovery corpus. Qwen tokenization,
source-group splitting, duplicate/label review and the number fitting the
416-token geometry remain to be established. No sealed evaluations were opened.

The new runtime has not trained the actual 3B model. Its GPU-memory fit, teacher
placement, optimizer state/offload, generation-aware checkpoint selection and
full-vocabulary generation path still need qualification before a scientific
run. Captured rollout rounds currently use fresh optimizers and explicit offline
staleness limits; continuous adaptive on-policy training is not implemented.
There is no accuracy, inference-speed or deployment claim from these checks.

The existing progressive run briefly stopped between phases when Infisical was
unreachable. After `inference-ctl whoami` succeeded, it resumed at prealignment
through new reservations with unchanged producer/configuration. The independent
all-36 run continued throughout. Their sparse-position results remain separate
from this new complete-response workflow.
