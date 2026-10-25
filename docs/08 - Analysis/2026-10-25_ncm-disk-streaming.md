# NCM bounded disk streaming

The existing Structure Net lossless feature loader supports this workload.
All 732 complete TRAIN feature tensors replayed with their original FP32 byte
hashes intact through a 256MiB retained-feature cache. No new loader, clipping,
quantization, source selection or model change was needed.

The replay read 18,719,846,400 bytes of decoded features (17.43GiB in total).
Peak retained features were 268,345,344 bytes, below the 268,435,456-byte budget.
The first and longest examples also reloaded exactly after eviction. Whole
process peak RSS was 1,315,291,136 bytes; the cache budget does not cover the
runtime, decompression temporaries or OS page cache. The first pass took
312.57 seconds alongside training, so this is not a controlled speed benchmark.

The original-runtime two-update disk pilot passed real CUDA backward/gradient
parity, the frozen-codec check and strict model/optimizer/scaler/RNG/output
reload. Peak GPU allocation was 2,264,888,832 bytes and reservation was
2,348,810,240 bytes. Whole process peak RSS was 5,631,684KiB (5.37GiB), including
CPU optimizer state, initialization and checkpoint copies. Reducing the feature
cache does not eliminate those other CPU allocations.

Compared with the completed large-cache pilot, the initial model and all 16
diagnostic hard-code and word-ID outputs matched. There were small differences
in 218 final model tensors, maximum absolute difference 8.1584e-6. Their cause
was not isolated; concurrent GPU execution and floating-point behavior remain
possible factors. Exact input bytes are established; bitwise training-trajectory
equivalence is not. These systems controls do not measure held-out accuracy.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_sequential_full732_disk_replay_v1`: full payload replay and raw hashes.
- `20261025_sequential_full732_disk_pilot_v1`: CUDA pilot and process resource log.
- `20261025_sequential_full732_disk_comparison_v1`: output and tensor comparison.

The old full training worker subsequently disappeared with its trace at
1718/2048 and no final checkpoint. That incomplete run is preserved. Future
launches use the bounded cache through the
[Pixi migration](../07%20-%20Status%20Reports/2026-10-25_ncm-pixi-migration.md),
with a separately qualified runtime and new artifact roots.
