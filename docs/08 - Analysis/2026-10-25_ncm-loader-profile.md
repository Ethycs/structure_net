# NCM loader preparation is serial

The current bounded loader has no background prefetch, pinned-memory staging or
batching of source examples. It loads and validates one example before the GPU
can process it. The training objective packs the twelve concept levels together;
gradient accumulation over four examples still executes those examples serially.
Structure Net has generic DataLoader/prefetch utilities, but this NCM-specific
full-source loader is not connected to them.

A CPU-only profile replayed the actual 32 presentations from updates257–264
through the existing 256MiB loader, while the GPU training run continued. All
32 were cache misses. Total time was 7.664s, or 0.958s per four-example update.
The retained-feature cache peaked at 267,239,424 bytes, below its bound.

| Operation, exclusive profiler time | Seconds |
| --- | ---: |
| SHA-256 constructors and updates | 2.917 |
| NumPy copies and conversion to bytes | 2.427 |
| zlib decompression | 1.621 |
| Tensor finite checks and reduction | 0.512 |
| Buffered file reads | 0.067 |

Linux reported 358,373,598 bytes read through the file API and zero additional
physical storage bytes for this process. These files were served from the OS
cache. This probe therefore identifies CPU preparation overhead, not an SSD
bandwidth limitation. It is not a controlled cold-cache or end-to-end benchmark.

For the common update256–1536 interval, the earlier large-RAM-cache run averaged
1.558s/update and the Pixi disk-streamed run averaged 2.594s/update. The observed
difference is 1.036s/update (66.5% more time); runtime and concurrent load also
differed, so this is not an isolated causal estimate. The independent loader
profile is consistent with preparation accounting for much of the difference.

The next optimization to test is a bounded background queue that overlaps full
example decoding/validation with model computation and retains exact source
bytes and sample order. Pinned staging and nonblocking GPU transfer can follow
if measured transfer time warrants them. Keep verification; do not trade away
hash checks or FP32 fidelity for a throughput claim. Reusing a raw lossless
mapped format is another option if decode/copy overhead remains dominant.
Any changed training loader needs a fresh lifecycle/parity pilot and artifact
root. The nearly completed active training run remains unchanged.

Evidence: `data/experiments/ncm_paper_regime/20261025_ncm_disk_loading_profile_v1`,
including per-example timings, a cProfile artifact, all function timings and
runtime metadata. No held-out quality or application tokens/s was measured.
