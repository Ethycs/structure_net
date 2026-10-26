# Original-body extraction pilot for the public source pipeline

Use all nine sources from the completed public source-metadata pilot, in its
fixed order. Load the unchanged original28-layer Llama3B body on physicalGPU0
RTX3060 in BF16, TF32 off, and reuse the existing single-example extraction and
FP32 character-alignment functions. No target field reaches the model. The
body and original MLPs remain frozen; no gradient or optimizer step is allowed.

First reproduce the saved first/longest historical TRAIN feature tensors with
the existing bitwise parity check. For each public pilot source, extract twice
and require equal FP32 bytes. Serialize using the already qualified lossless
byte-shuffled zlib format, reload through the original binding validator, and
require the decoded bytes to match exactly. Save source/prompt/alignment/body
bindings, payload hashes, elapsed extraction/storage times and GPU allocation.
Require unchanged body hash and peak allocation <=8,600,000,000bytes.

Fresh root:
`data/experiments/ncm_paper_regime/20261026_public_feature_pilot_v1`.
This is a systems pilot, not whole-corpus feature admission or quality evidence.
The full metadata pass has not completed yet; its actual longest inputs must
be qualified separately before a whole-corpus streaming launch. Do not
materialize an eager whole-corpus feature cache. The nine small pilot payloads
are retained as auditable controls for the later bounded disk buffer.

Run from the repository root with locked Pixi:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_public_feature_pilot run
```

Run existing lossless feature-codec and source-only metadata tests first.
Preserve failures and actual terminal receipts. No final source, model-quality
evaluation or judge request is involved.
