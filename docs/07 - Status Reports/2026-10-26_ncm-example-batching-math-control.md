# Attention backend control after the default batching tolerance failed

The completed default-backend probe from commit92b45dc failed its clean FP32
comparison: relative gradient error8.359307106449387e-5 exceeded5e-5, and
logit allclose(rtol=atol=1e-4) failed. Maximum absolute gradient error was
4.174187779426575e-6; maximum logit difference was0.0008144378662109375.
All1282 code argmaxes matched. Actual session29714 exited0, chunkf07d30.
No noisy comparison, capacity sweep or optimizer step ran. The model tensor
hash stayed unchanged. This failed gate remains failed.

This prospective diagnostic changes only the attention backend used in the
FP32 equivalence comparison to PyTorch SDPA MATH. Batch-shaped CUDA arithmetic
is a hypothesis for the discrepancy, not an established diagnosis. Reuse the
same shortest/longest pair, histories, checkpoint, gradients and all original
tolerances. Test clean then noisy histories; stop at the first failure. A pass
would qualify the batching objective under this reference backend, without
claiming that the original default backend passed its tighter test.

If both reference comparisons pass, run the originally specified FP16
resource sweep with default backend selection. Resource qualification is not
numerical equivalence, full optimizer lifecycle qualification, application
speed, or semantic quality. No optimizer step is allowed. The unchanged 95%
complete-summary task goal and all data restrictions remain in force.

Use a fresh output root
`data/experiments/ncm_paper_regime/20261026_example_batching_math_probe_v2`.
Producer copies include this amendment and both launchers; original evidence
is append-only. Launch from the repository root after saving this producer:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_example_batching_math_probe run
```
