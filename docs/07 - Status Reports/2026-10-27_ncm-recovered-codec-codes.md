# Fixed code targets for recovered-domain codec training

Encode every one of the 99,367 prepared native text frames with the unchanged
codec encoder and codebooks. This is representation-training preparation, not
source-to-summary supervision or a student-quality result. The 128 diagnostic
source components and their exact native frames remain excluded.

Use the original released codec, whose encoder has already matched the current
repaired codec on every diagnostic target. Its expected full tensor identity is
`9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32`.
Use deterministic FP32, TF32 off, locked Pixi on GPU2 RTX2060SUPER, batch eight,
with a 3.5 GB own allocation cap. Before full encoding, compare batch-eight and
individual encodes exactly on 32 fixed TRAIN frames: first eight, last eight,
eight longest and eight shortest, filling duplicate selections in source order.
Preserve a failed parity check; do not silently change batch size or precision.

Read the input uint16 binary through a read-only memory map. Write 641 uint16
codes per frame to disk, with ordered offsets, frame hashes and code hashes in
an index. Validate range and exact batch coverage and read every output record
back before completion. Individual/batched parity is measured on the 32 controls,
not asserted for every untested frame. Keep the encoder/codebooks unchanged and
all gradients absent. Do not load the Llama body or disturb current GPU workers.

Use `20261027_recovered_codec_codes_v1` for append-only receipts and
`/data2/structure_net/ncm_recovered_codec_20261027_v1/codes` for binary storage.
Freeze source, input and environment identities before inference. There are no
optimizer updates, teacher requests, semantic labels admitted, or final accesses.
