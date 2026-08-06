# NAL Local GPU Scheduler Acceptance

**Status:** ACCEPTED FOR LOCAL EXPERIMENT PARALLELISM

**Date:** 2026-08-06

**Architecture:** `../03 - Architecture/nal-local-gpu-scheduler.md`

## Verdict

The canonical local NAL runner now has a tested multi-GPU and multi-slot
contract. It uses spawned workers, logical CUDA ordinals, bounded retries, and
fingerprinted completed-experiment resume. Ray remains outside the accepted
path.

## Hardware evidence

On a PCI-pinned 8 GiB NVIDIA GeForce RTX 2060 SUPER:

1. Free-memory calibration observed 7.51 GiB available and launched three tiny
   TinyLLM seed processes on three slots of one logical GPU.
2. Seeds 7, 17, and 29 completed concurrently. All nine matched arms shared
   their per-seed initialization and restored every checkpoint exactly.
3. Re-launching the identical campaign returned all three completed results
   from the ledger without starting workers.
4. With physical GPUs 0 and 2 exposed together, NAL mapped them to logical
   devices 0 and 1 and completed one TinyLLM seed on each concurrently. Both
   passed matched initialization and exact checkpoint restoration.
5. Three actual TinyLLM `d6` jobs, each with 30,345,602 parameters after
   feedback growth, trained concurrently on the same card under the automatic
   1.5 GiB-per-seed estimate and three-slot cap. All completed every control and
   exact checkpoint restoration without an out-of-memory error.
6. Each `d6` process measured approximately 0.653 GiB peak allocated and at
   most 0.711 GiB peak reserved CUDA memory for this short sequence/batch
   profile. The 1.5 GiB admission estimate intentionally retains more than a
   two-times margin; longer sequences or larger batches require a new pilot.

These are scheduler and lifecycle results. The synthetic loss differences are
not evidence that feedback improves language modeling, and concurrent wall
times are not performance benchmarks.

The final repository gate completed with **384 passed, 1 skipped, 0 failed**
and 23 warnings in 351.19 seconds. The skip is the hardware-gated CUDA test in
the sandboxed suite; the same CUDA lifecycle was exercised directly on the
host hardware described above. The focused scheduler/configuration/campaign
gate completed with **37 passed, 1 skipped, 0 failed**.

## Acceptance matrix

| Claim | Result |
| --- | --- |
| No implicit physical GPU masking | Accepted |
| Logical-ID validation after visibility mapping | Accepted |
| Fixed per-GPU slots | Accepted |
| Free-memory-derived slots with cap | Accepted |
| CUDA-safe process creation | Accepted with `spawn` |
| Several seeds sharing one GPU | Accepted on real hardware |
| Failure retry | Accepted in focused tests |
| Fingerprinted completed-result resume | Accepted in tests and real relaunch |
| Isolated timing separation | Accepted in output schema and CLI |
| Ray equivalence | Not claimed |
| Multi-node or DDP | Not claimed |
