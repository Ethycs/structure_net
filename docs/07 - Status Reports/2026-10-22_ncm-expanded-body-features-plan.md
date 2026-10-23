# Frozen parent features for broader NCM training

Date: 2026-10-22. Training-data preparation only, no task-accuracy hypothesis.
The existing8,192-source teacher job remains live and immutable. Prepare features
from its already frozen public sources independently of label arrival or quality.
This is an offline training optimization; it does not remove the parent forward
pass from inference and cannot establish a speedup.

Use the exact cached Llama-3.2-3B-Instruct model, all28layers frozen, BF16 SDPA,
physicalGPU0, batch4, seed42. Preserve original prompt IDs,512-token input canvas,
attention masks and explicit position IDs. No truncation, new source selection,
teacher requests, original-body training or generation. Verify original body
digest `75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c`.

A real lifecycle pilot first recomputes the original first four training cases
and requires exact last-valid-state parity with their existing feature cache.
It also processes the first four fresh training sources, preserves model state,
and reloads the saved feature bank exactly. Full preparation then processes all
8,192 fresh sources, in original order, only after a matching passed pilot.

Store each source's valid hidden states, with padding removed, as its original
BF16 tensor. Also save a small row-aligned bank of last valid states for the
current global residual adapter. Keeping token states permits later source
alignment experiments without repeating the original body forward. No tensor
quantization or learned feature compression in this preparation. Save source,
prompt and output hashes, memory peak, order, and processed-count progress.
Stop on OOM, nonfinite states, changed inputs, identity/parity failure or invalid
source length; no silent batch reduction or CPU fallback. Preserve any failure.
Keep resident services and all original experiment producers unchanged.

Artifacts: `20261022_expanded_features_{pilot,train}_v1` under the paper root.
Feature production never admits a model or noisy label. The separate full
teacher audit, development-quality gates,128-token summary cap and sealed final
test policy remain unchanged.
