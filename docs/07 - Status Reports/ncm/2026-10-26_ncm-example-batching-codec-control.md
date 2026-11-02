# Frozen codec preparation control after both attention backends failed parity

The default and MATH attention backend comparisons both failed the original
strict FP32 equivalence gate, with respective relative gradient errors
8.359307106449387e-5 and8.35528582466022e-5, and zero argmax differences.
Changing the attention backend alone did not resolve the discrepancy.
MATH session39255 exited0, chunkdfb491. Both failures remain append-only.

This prospective third diagnostic tests whether preparing frozen residual VQ
conditioning separately for each example resolves the discrepancy. The
released codec includes convolutional residual contributions and an accumulated
reconstruction. Batch-dependent rounding there is a hypothesis. Measure the
clean conditioning tensors produced separately versus together, then preserve
the separate computation and concatenate before calling the unchanged batched
generator and source attention. Do not alter any checkpoint tensor, precision,
data, code target, attention backend or numerical threshold.

Repeat the original clean/noisy FP32 two-example gradient comparison. If either
fails, stop. If both pass, run the original candidate1,2,4,8 FP16 resource sweep
using the same per-example codec preparation. This is a different execution
variant; a pass cannot retrospectively rescue either previous failed variant.
Resource timings exclude optimizer, feature extraction and disk loading.
No optimizer or quality qualification is part of this diagnostic.

Use fresh root
`data/experiments/ncm_paper_regime/20261026_example_batching_serial_codec_probe_v3`.
The frozen implementation, tests and unchanged-weight safeguards of the first
probe remain applicable. Run from the repository root with locked Pixi:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_example_batching_codec_probe run
```
