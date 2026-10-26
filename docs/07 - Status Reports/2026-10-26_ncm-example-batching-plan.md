# Example batching qualification before larger conditional training

This prospective systems diagnostic, `ncm-example-batching-v1`, asks whether
complete examples can be batched without changing the teacher-forced objective,
and which tested sizes fit physical GPU1 RTX2060. It is
`systems_lifecycle_only_not_quality_evidence`. It neither changes the 95%
complete-summary semantic goal nor admits public training data.

Use the unchanged BROAD checkpoint (file SHA256
`74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff`),
1376 previously admitted TRAIN cases and their original lossless FP32 source
features. No development or final source is read. The codec and word adapters
remain frozen. No optimizer step or new learned checkpoint is produced.

First test collation, padding isolation, source detachment, uneven microbatch
weighting, and example-then-level corruption RNG order on CPU. For real-model
FP32 gradient parity, compare two separate examples with one padded batch of
two: the shortest and longest source, sorting ties by ID. Disable dropout,
retain trainable generator/source-adapter gradients, and test both clean and
corrupted histories (seed 212142). Require every gradient presence to match,
relative gradient L2 error <=5e-5, maximum absolute gradient error <=1e-4,
logits allclose rtol/atol 1e-4, and no argmax disagreement. Fixed matching
histories are supplied to both executions. Retain failure evidence and stop
capacity testing if either parity comparison fails.

After parity passes, test FP16-autocast forward/backward with FP32 weights and
FP32 cross-entropy, dropout active, and a fixed effective batch of eight.
Use a fixed loss scale128 and unscale FP32 gradients before checking finiteness.
Use the eight longest distinct TRAIN sources ordered by descending length and
ID. Candidate microbatch sizes are 1,2,4,8. For each size, perform one warmup
and three measured effective batches, zeroing gradients between effective
batches. Use original per-example noise order with corruption seed 213142,
and dropout seed 214142 reset for each candidate. No optimizer is created.
All sources and all641 target codes stay complete. Store each elapsed time,
finite loss/gradient checks, actual CUDA allocated/reserved peaks and host
feature-cache metrics. Stop larger sizes at the first CUDA OOM or peak
allocation over4,900,000,000bytes. Never retry a failed size in place.

These timings measure forward/backward with already-loaded source features,
including collation/transfers/noise and finite-gradient checks. They exclude
body extraction, disk loading and optimizer work. They cannot establish
application inference speed, overall training speed or model quality.
Dropout draws and FP16 roundoff need not be identical across batch shapes.
FP32 parity covers a batch of two; larger sizes receive resource/finiteness
qualification only and need additional numerical and full optimizer lifecycle
qualification before a larger training run adopts them.

Use locked Pixi `ncm` from the repository root. Launch after focused tests and
producer freeze:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_example_batching_probe run
```

Fresh output root:
`data/experiments/ncm_paper_regime/20261026_example_batching_probe_v1`.
Record configuration, dependency/input hashes, copied producing sources,
environment, raw metrics, final unchanged-model hash, actual terminal receipt
and a conservative report. Any launcher/code correction requires a new root
and an explicit amendment preserving the original failure.
