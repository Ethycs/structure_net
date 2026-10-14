# Frozen Llama 3B held-out comparison

Date: 2026-10-13. Parent `llama32-3b-summary-throughput-v1`.
No held-out model outcomes have been generated or inspected.

Select the existing Q4_K_M checkpoint and draft length 8, using shared prompt v3.
The 16-request development screen produced correctly attributed final decisions
with no extra summaries in both ordinary and speculative Q4 decoding. Locally
converted Q4 failed two large batches through repetition and fragment mixing and
is rejected. Retain its complete artifacts; do not pool it with the selected model.
FP16 development throughput was 33.78 output tokens/s; selected Q4 draft 8 was
80.65 tokens/s (2.39x). These are selection data, not primary evidence.

The pinned held-out fixture SHA-256 is
`8498b1597132f41fd0509e8cdc09dae0423a1be11f3ef75e3216e3cf58636d11`.
It contains 20 requests from five new seeds across all four input families, with
55 source fragments. Prompt lengths are 496–6,295 tokens. Both raw 768-token and
hierarchy 512-token generation caps and natural EOS stopping remain unchanged.
Select nothing further using held-out outcomes; any subsequent tuning needs a
new study and new held-out inputs.

Primary order is FP16 A, existing Q4 A, existing Q4 B, FP16 B, with two repetitions
in A and three in B: five observations per request and strategy, 400 requests
total. Within each process, draft 0 and draft 8 alternate order by repetition.
All arms use the same full 28-layer model family, native runtime, FlashAttention,
F16 KV storage, 8,192-token context and GPU 0. No parallel GPU experiments.
Primary baseline is optimized FP16 ordinary generation. Existing Q4 ordinary
generation remains a separately reported control. This is not a claim of twofold
speed over the already quantized deployment.

Source-grounded quality review must cover every distinct output from every arm,
including attribution, updated facts, invented material, missing/extra fragments
and the 128-token stored-summary ceiling. A request fails if any observed output
variant fails. Review is performed by the coding assistant with access to sources,
not a blinded human or an independent benchmark evaluator. These synthetic,
repetitive final-decision sources favor prompt lookup; do not generalize their
success rate or speed to production summarization or mobile hardware.

After primary timing, run separate full-capacity checks with EOS stopping disabled
at the raw 768 and hierarchy 512 caps. Those establish runtime capacity only and
do not contribute to quality or primary throughput claims.
