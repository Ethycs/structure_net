# Require deterministic GPU arithmetic after the failed optimizer replay

The corrected four-example lifecycle v1 exited1 during exact resumed-update
verification. Both native-reference gradient comparisons passed: relative L2
errors6.0156e-6 clean and5.9002e-6 noisy, no argmax disagreements. Peak allocated
GPU memory was4,009,642,496bytes, below the unchanged4.9GB bound.

The original and replayed second update used identical sampling, corruption,
forward losses, scaler state and CPU/CUDA RNG states. Model and optimizer
states differed; gradient norms were3.4873967171 and3.4874844551. This suggests
nondeterministic backward arithmetic, but the responsible operator has not
been isolated. The original failed root, checkpoints, source and receipt remain
unchanged. This does not explain the older0/128 quality results or establish a
training improvement.

Repeat the same four cases, initializer, native-reference comparisons, two
effective8 updates, seeds, optimizer, precision,4.9GB allocation bound and exact
resume/output requirements in fresh root
`20261026_public_batch4_lifecycle_deterministic_v2`. The sole intervention is
explicit deterministic execution: `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
Python, `torch.use_deterministic_algorithms(True, warn_only=False)`, deterministic
cuDNN and cuDNN benchmarking disabled. PyTorch2.4 documents that deterministic
mode selects supported deterministic implementations or raises an error; the
flag alone does not guarantee whole-application reproducibility. The actual
replay remains the acceptance test.
[PyTorch2.4 documentation](https://docs.pytorch.org/docs/2.4/generated/torch.use_deterministic_algorithms.html).

Entry point: `experiments.structure_net.ncm_public_batch4_deterministic run`,
through locked Pixi ncm on physicalGPU1. Stop on any unsupported deterministic
operator, OOM, numerical/parity or replay failure. Do not relax equality,
tolerances or switch batch size within this run. No new semantic judgment,
final-test access or application-speed claim occurs. If this passes, the full
mixed-data pipeline must use the same execution settings and qualify its own
effective128 replay before larger training.
