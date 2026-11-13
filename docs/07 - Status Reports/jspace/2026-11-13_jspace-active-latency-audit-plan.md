# Matched single-label latency audit — exploratory follow-up

Designed after the active CAV/head study and its 2.505× task-latency result.
The original comparison completes a word plus EOS for the teacher but returns
a class index for the prefix. Preserve that measurement and isolate this
interface confound in a separate, post-hoc audit. No training or model selection.

Use the same frozen checkpoint, selected block-29 full linear head, and existing
64 evaluation sources. Before timing, require the teacher's first token to be a
single colour token matching its recorded completed answer on every source.
Compare three functions, each returning one class index after one prefill,
with cache disabled and no continuation:

1. Full 36-block model, final norm and full vocabulary argmax mapped to colours.
2. Full 36-block model and final norm, scoring only the four colour rows of the
   existing vocabulary head. This restricts the answer vocabulary; verify all
   64 decisions match the full-vocabulary teacher and count failures otherwise.
3. Actually shortened 30-block model, no final norm or vocabulary head, with the
   unchanged trained linear classifier. Verify all 64 frozen predictions.

Same first 32 source prompts, batch one, BF16 backbone, four warmups per arm,
five repetitions, CUDA synchronization around wall time including tokenization
and readout. Interleave the two teacher arms in seeded random order (941).
Then delete the six later blocks, norm and vocabulary head, release unused
allocator blocks, and measure the prefix. Report both median speed ratios,
all per-call times, unique backbone parameter counts and peak reserved memory.
Include the prefix's FP32 classifier in memory; preserve original tied input
embedding weights. All arms produce class indices on the host.

This remains a single-device, warm-weight, sequential teacher-then-prefix
benchmark on a reused synthetic task. It is not an independent accuracy trial,
a generic language-model benchmark, or generic tok/s. The four-row teacher is
the stronger matched baseline. No endpoint in the original study changes.

Freeze this protocol, producers, test and input identities before measurement.
Use root locked Pixi, CUDA hidden for preparation, and a live 10,240 MiB 3060
reservation before model loading or CUDA queries. Release on exit. Archive
strict JSON evidence and an append-only aggregate; do not start any service.
