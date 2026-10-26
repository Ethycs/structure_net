# Frozen codec targets for the full admitted public training catalog

Encode all 162,353 admitted original public reference frames using the exact
repaired codec in the BROAD initializer, hash
`eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366`.
This prepares supervised training labels, not source-only predictions or
quality measurements. The catalog's 320 reserved public development articles,
application protections and sealed final remain excluded.

Use physical GPU1 RTX2060, FP32, TF32 off, one complete native128 frame per
encode. Freeze all codec tensors. Require exactly 641 integer code IDs in
0..8191. Encode each selected frame twice and require byte-identical IDs;
retain unchanged codec hashes and no gradients. Reproduce all 16 existing
BROAD diagnostic TRAIN code hierarchies before public preparation.

The public pilot has 16 distinct catalog records: first row; nearest unused
native target lengths 1,16,32,48,64,96,126 (ID tie break); then smallest ID
hashes until 16. No output determines selection. After observed successful
pilot completion, process every catalog record in original order, requiring
all pilot targets to match exactly. Keep the same repeated-encode check on
every full-run record. Write indexed JSONL containing target/native-frame/
source bindings, the fixed codec hash and the entire code hierarchy.

A separate CPU audit must replay every pairing, byte offset and complete-code
shape/range against the admitted catalog before training. It verifies saved
GPU evidence and bindings, not an independent GPU re-encoding. The full run
may coexist with GPU0 frozen-body feature extraction: the two preparations
have no shared mutable model state. Keep CPU input buffering bounded through
the indexed disk dataset. Do not change root Pixi dependencies.

Evidence roots: `20261026_public_code_targets_{pilot,full}_v1`. Run from the
repository root through locked Pixi `ncm`. No training update, new judge,
complete-summary accuracy or application speed measurement occurs here.
