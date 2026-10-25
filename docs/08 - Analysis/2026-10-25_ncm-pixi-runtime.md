# NCM now runs through Pixi

The named `ncm` environment is installed at `.pixi/envs/ncm`. Its package
requirements are declared in `pyproject.toml`, with transitive dependencies and
package hashes in `pixi.lock`. Execute from the repository root with
`pixi run --locked -e ncm`. No new command uses the MoBA interpreter under data.
The default environment's package selections and original package hashes were
checked against the previous lockfile and are unchanged.

The old interpreter was a `--system-site-packages` virtual environment layered
on the default Pixi Python. It originated in the MoBA FlashAttention compatibility
experiment; carrying it into NCM was a workflow error. The new launcher also
resolves Hydra, OmegaConf, tiktoken and pysbd inside Pixi, avoiding the old
dependency overlays in data. Existing downloaded model source, inputs and run
evidence stay at their immutable artifact locations.

Qualification completed:

| Check | Result |
| --- | --- |
| Dependency consistency (`pip check`) | No broken requirements |
| Focused NCM tests, through the Pixi task | 10 passed |
| Stored TRAIN target token frames | 732/732 identical |
| Saved initial rendered outputs | 16/16 identical |
| Actual RTX2060 FP16 forward/backward | Finite |
| Two-update model pilot | Exit 0 |
| Packed/sequential gradient parity | Passed |
| Frozen codec and strict checkpoint/state/output reload | Passed |
| Whole-lifecycle peak GPU allocation | 2,264,888,832 bytes |
| Whole-lifecycle peak GPU reservation | 2,348,810,240 bytes |
| Feature cache | Existing lossless disk LRU, 256MiB limit |

The initial test command imported unrelated global network-factory fixtures and
failed on missing SciPy. The corrected task scopes discovery to the NCM tests;
none of these tests use those fixtures. Both the initial failure and successful
task are recorded. A second tokenizer qualification freezes the corrected
manifest. This is a runtime migration, not a held-out quality result.

Python, torch, CUDA, NumPy and Transformers retain their intended versions.
The released tokenizers 0.21.4 replaces the inherited development build, and
Triton 3.0.0 satisfies torch 2.4.1's requirement instead of the inherited 3.1.0.
The input-ID and rendering checks cover this tokenizer change. Complete package
versions/import locations and manifest/lock hashes are saved with the producer.
No bitwise training-trajectory equivalence is claimed.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_ncm_pixi_runtime_v2`: runtime and tokenizer qualification.
- `20261025_ncm_pixi_systems_v1`: CPU tests, CUDA smoke test and prior-worker state.
- `20261025_sequential_full732_pixi_pilot_v1`: real model pilot and terminal receipt.

The earlier full training run has no live worker or final checkpoint; its trace
ends at 1718/2048. Its cause and exit code are unknown. Preserve the incomplete
root. A replacement fixed-budget run must start from the original qualified
parent, use the new Pixi root, and be reported separately. The completed pilot
now permits `pixi run --locked -e ncm ncm-train` under the
[migration amendment](../07%20-%20Status%20Reports/2026-10-25_ncm-pixi-migration.md).

After a successful full-run terminal receipt, the existing semantic gates are
available through `experiments.structure_net.ncm_pixi_development`, in order:
`inference_pilot`, `outputs`, `judge_pilot`, `grade`. Select physical GPU1 before
the inference commands. This wrapper points to the new checkpoint and fresh
DEV roots, preserving the source split, rubric, judge identities and historical
first judgments. It has not yet run. Final320 and speed remain gated on quality.
