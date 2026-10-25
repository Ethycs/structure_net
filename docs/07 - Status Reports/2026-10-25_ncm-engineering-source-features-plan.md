# Frozen-body engineering features without label access

Extract all 888 prepared source inputs (732 TRAIN, 156 engineering heldout)
using the original 28-layer Llama body and all original MLP weights unchanged.
Use physical RTX3060 GPU0, original BF16 weights/computation, TF32 off and the
same complete character alignment to FP32 GPT2 source positions. Parent inputs
contain only the source prompt; no summary or target code is present.

First run the first/longest-GPT2/longest-native source probes (unique, at most
three) with repeated extraction and exact original TRAIN reference parity. Check
all frozen body tensors by the established body hash. Compress each FP32 feature
tensor with the existing audited lossless byte-shuffled zlib1 format and verify
every stored roundtrip byte-for-byte. A successful real CUDA pilot admits the
full cache, preserving pilot entries exactly.

The uncompressed feature estimate is 21,323,083,776 bytes. Current free disk is
67 GB; keep an 8 GiB floor, with no source truncation or dtype fallback. Original
GPU peak limit is 8.6 billion allocated bytes. Snapshot code, tokenizer metadata,
prepared input hashes and unchanged parent weight files into new stage producers.
Supervise child processes with actual exit receipts. No generator/word decoder,
target codes, model updates, new judge calls or sealed final source access.

The resulting cache supports the next fixed-budget attention-code-repair
experiment. Frozen feature extraction is not training, task accuracy, or an
inference speedup measurement. The complete-summary semantic 95% goal remains.
