# Both RTX 2060s for expert banks

Exploratory hypothesis `word-games-three-gpu-experts-v1`, 2026-10-31.
User explicitly proposed temporarily stopping Triton Qwen3-8B to host
experts across both RTX 2060s. This supersedes the prior prohibition on
unloading that model for this experiment only. Unload only that model via
Triton's repository API, leave OCR and other services running, and reload
the identical model in a finally block. Verify model readiness afterward.

Known before this test: the custom fine-grained 64-pin split completed
its three-request pilot at 24.27 end-to-end tok/s, versus a matched whole-bank
eight-layer control at 30.45. GPU hits were 25.7%; the split retained all
selected experts and passed its numerical checks but did not pass the
competitive-speed gate. Preserve this negative pilot; do not advance it.

Use original pinned native b11960, unchanged Empero 35B Q4_K_M weights,
Q8 KV, Flash Attention, 12 CPU threads, and the existing frozen prompts.
All attention, recurrent state, dense/shared work and upper 14 expert banks
stay on RTX 3060. Logical CUDA1 is RTX 2060 SUPER, CUDA2 is RTX 2060 6 GB.
Baseline: eight banks (layers 0-7) on SUPER, layers 8-25 on CPU.
Candidate: keep those eight banks on SUPER, put ten more banks (8-17) on
the temporarily freed 2060, leaving layers 18-25 on CPU. Tensor overrides
must match every gate/up/down expert tensor and all 42 layer assignments
must remain on CUDA0. These are intact banks, not expert pruning or another
precision. No custom hybrid callback is enabled in this experiment.

Validate placement and memory with the three-request pilot for both arms.
Stop only the experiment container if any GPU free memory falls below
384 MiB or the Triton server loses readiness twice. Expect Qwen3-8B itself
to be temporarily unavailable. If the candidate cannot fit, record that
failure and use eight additional banks in a separately identified arm.
Advance only a candidate at least 5% faster in total pilot completion time
with valid, untruncated JSON. Run one full 36-request replay of baseline
and candidate; report descriptive paired timing, not a confidence interval.
If a full pass is invalid, stop rather than treating a shorter response as
speed. Full quality diagnostics cover receiver state/decisions, sender
structure, and manual authorized-content review. No population accuracy or
semantic noninferiority claim follows from a single fixed suite.

Fingerprint sources, runtime, checkpoint, input corpus, exact commands,
placement logs, response hashes and service lifecycle records. Keep raw
evidence append-only in dated roots. Register the conservative stabilized
result and verify exact and semantic read-back. Service restoration is a
required final action even if model loading or timing fails.
