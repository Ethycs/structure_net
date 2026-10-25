# NCM Pixi runtime migration

This is a systems amendment to the existing full732 experiment. The user
rejected execution through `data/experiments/moba_paper/environment/bin/python`.
Use `pixi run --locked -e ncm` from the repository root. Dependencies belong in
the named `.pixi/envs/ncm` prefix, declared in `pyproject.toml` and resolved in
`pixi.lock`. Preserve old environments and frozen run evidence for provenance.

The historical environment was an overlay on the default Pixi interpreter,
originally made for a MoBA FlashAttention ABI requirement. Reusing that overlay
for NCM was a workflow mistake. The NCM path on the RTX2060 uses PyTorch SDPA;
it does not require that FlashAttention wheel or torchvision. The named NCM
environment excludes the default feature and does not reinstall the default
environment. Its additions to the lockfile must preserve default package pins.

Preserve Python 3.11.13, torch 2.4.1+cu121, numpy 1.26.4 and transformers 4.53.3.
Use released tokenizers 0.21.4 in place of the inherited 0.21.4.dev0, and torch's
resolved Triton 3.0.0 dependency in place of the inconsistent inherited 3.1.0.
Record all installed package versions and import locations. Use Pixi packages
for Hydra, OmegaConf, tiktoken and pysbd, rather than their legacy dependency
directories in data. Keep the immutable downloaded upstream model source and
checkpoints in their existing artifact locations. No model code is rewritten.
Keep the legacy launch's unset CUBLAS_WORKSPACE_CONFIG; do not claim bitwise
training equivalence across runtimes or GPU scheduling conditions.

Known at this amendment: the full732 v2 trace stops at update 1718/2048. The
recorded worker PID is absent, no matching worker is live, the old tool session
is unavailable, and no final checkpoint or result exists. The cause and exit
code are unknown. Preserve that incomplete root; do not manufacture a success
receipt or treat its partial losses as accuracy. The loader-only full732 replay
and the earlier disk lifecycle pilot completed successfully before migration.

Qualification order:

1. Install only `ncm`; run `pixi run --locked -e ncm python -m pip check` and
   `pixi run --locked -e ncm test-ncm`.
2. Run `pixi run --locked -e ncm ncm-check`. Verify all 732 stored target token
   frames and saved initial renderings using the released tokenizer. Record
   package locations inside the Pixi prefix and a hash of manifest and lock.
3. Run `pixi run --locked -e ncm ncm-pilot` on physical GPU1. Reuse the original
   two-update objective/gradient-parity, memory and strict checkpoint-reload
   lifecycle, with the existing lossless 256MiB disk LRU. Preserve complete
   sources, all seeds, optimizer placement, precision and allocation limits.
4. Only a successful terminal pilot permits `pixi run --locked -e ncm ncm-train`.
   Start from the same qualified decoder and generator parent for the unchanged
   fixed2048-update budget, because the interrupted run has no resumable state.
   Do not combine its partial updates with the new run's counters.

The first CPU test invocation found that the repository's global conftest imports
unrelated network-factory/SciPy components. The focused NCM tests use no fixtures
from that conftest; scope their discovery with `--confcutdir=tests/structure_net`.
All ten pass with that scope. Preserve the original tokenizer qualification as
runtime_v1 and repeat it as runtime_v2 after correcting the manifest's test task.

New artifact roots are `20261025_ncm_pixi_runtime_v2` and
`20261025_sequential_full732_pixi_{pilot,train}_v1` under the existing NCM root.
The launcher reuses the frozen training implementation through process-local
configuration and records the effective runtime plus manifest/lock hashes.
After training, adapt the existing source-disjoint DEV evaluation to this
checkpoint under a fresh root. Semantic quality remains unmeasured until that
evaluation completes. The final320 and speed test remain behind the original
quality gates. Environment and loader checks cannot establish task accuracy.
