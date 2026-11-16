# Progressive response rebuild: capture systems amendment

Parent: `JSPACE-QWEN3B-PROGRESSIVE-RESPONSE-REBUILD-01`,
[prospective protocol](2026-11-16_qwen3b-progressive-response-rebuild.md).

The20261116_v1 capture did not produce scientific data. Its first reservation
request failed on a transient Infisical timeout, before any worker launched.
The next request received3060 lease `b5d096cdd2f7`; loading the native teacher
then failed before the first record with `Inference tensors do not track
version counter`. The capture constructed model tensors inside
`torch.inference_mode()` and subsequently attempted the declared `_version`
immutability check. Preserve both controller failures and the native failure
receipt; no trained checkpoint or quality outcome exists for this attempt.

Correct model construction to keep ordinary versioned tensors while retaining
gradient-free capture and the original immutability check. Do not waive the
check, alter teacher weights, change precision or relax bit-exact native seam
replay. Add a CPU regression covering the construction context. The new source
is archived under `jspace_3b_progressive_response_rebuild/20261116_v2` before
GPU execution. Corpus, selections, probes, compression dimensions, fitting
order, recovery budget and advancement gates are unchanged.

The original progressive worker already stopped cleanly after saving and
evaluating update90. Reuse the authenticated v1 transition receipt; do not send
another signal. The all-at-once worker and its2060SUPER lease remain untouched.
