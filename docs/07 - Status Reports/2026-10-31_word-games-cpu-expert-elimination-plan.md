# Remove the last CPU expert banks

Prospective exploratory study `word-games-cpu-expert-elimination-v1`.
Parent: the completed packed38 and sampling-output experiments. No production
promotion. Preserve existing worktree changes and completed evidence.

Inventory before this plan: layers 24 and 25 hold 432 and 498 MiB of routed
expert projections. All other serving expert banks already reside on GPUs.
Recent packed38 control minimum free memory was 961 / 597 / 875 MiB on
3060 / SUPER / regular 2060, with a 384 MiB guard. This is native C++ CPU
backend work, not Python workers or the earlier separate draft-model service.

Use the verified patch-1 runtime, with the slower grammar-preview patch disabled.
Keep the 35B Empero checkpoint, all 256 experts/top-8 routing, attention and
shared computation on 3060, Q8 KV, Flash Attention, context 8192, batch/ubatch
512, frozen 36 requests, temperature zero, seed 73 and no prefix reuse.

Arms fixed before new timing:

- `packed38`: unchanged weights and placement, CPU banks 24/25, 12 threads.
- `gpu40`: unchanged weights; move 25 to 3060 and 24 to regular 2060, 12 threads.
- `gpu40_cpu1`: same all-GPU weights/placement, one decode and batch CPU thread.
- `q2_3060`: requantize only the six routed projections of 24/25 to Q2_K;
  put both banks on 3060; use one decode and batch CPU thread. All other tensor
  payloads must be byte-identical. This is lossy weight compression, not a
  lossless transformation. The two banks shrink from 930 to 504 MiB. Q3_K
  would need 660 MiB and misses the current 3060 headroom guard without moving
  other work, so it is not the selected local-placement arm.

First audit exact tensor sizes and sample reversible compression of the stored
quantized bytes; label sampled compression ratios as estimates, not a GPU codec.
Prepare a new derived checkpoint without overwriting the original. Verify all
untouched tensor bytes and changed tensor identities, and check tokenizer,
template and model metadata. Source/build/runtime stay outside data; raw
evidence and derived checkpoint files are data, never executable environments.

Pass focused configuration, placement and GGUF parsing checks plus CUDA
availability before startup. Freeze/hash the producer and all inputs. Use new
`data/experiments/word_games/20261031_cpu_experts_*` artifact roots.

Run the existing three-request pilot in baseline then candidate order. Advance
each candidate to two full replays if all pilot requests complete as valid,
untruncated JSON and its token throughput is at least 90% of the baseline.
This is a measurement campaign, not a prospective speed-win gate. Full order
is baseline, gpu40, gpu40_cpu1, q2_3060, then reversed, omitting failed arms.
Stop a failing arm on runtime/placement failure, OOM, less than 384 MiB free
on any used GPU, invalid output or service failure; preserve its failure.
Do not lower the memory guard or silently change placement/precision.

Report visible token and byte throughput, request/prefill/decode times, GPU
headroom, host RSS and server process user+system CPU seconds. Process CPU
sampling surrounds each complete call and therefore includes template handling;
the established request wall timer measures completion only. It is not a
kernel profile or an isolated measurement of the two CPU banks.

Check receiver decision/occupied-slot correctness against each authorized
packet, selector schema validity, and sender structure agreement separately.
Compare exact outputs but do not equate wording differences with quality loss.
Review receiver summaries against their packets, retaining known baseline
unsupported-approval defects. No population accuracy or independent-judge
claim. Requantization or CPU/GPU arithmetic may change generated outputs.

The regular 2060's Qwen3-8B may be temporarily unloaded under the existing user
authorization. Wait for two release samples, keep OCR resident, and restore
Qwen3-8B readiness/configuration in finally. Register and read back a conservative
completed result, report all attempts, and commit the durable source/docs.
