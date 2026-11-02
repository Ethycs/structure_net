# NCM trained on a frozen Llama body

Date: 2026-10-22. Hypothesis: `ncm-frozen-llama-body-v1`.
Exploratory, one seed; not a confirmatory architecture comparison.

Question: can the NCM generator learn to summarize from a frozen Llama-3.2-3B
body without updating the body or the existing codec? Does this improve complete
summary quality over additional matched training of the native NCM?

The earlier standalone result is known: 5/256 LLM-judged task passes. The shared
saved targets were often supported snippets rather than main-point summaries.
That evidence and its completed artifacts remain unchanged.

## Fixed design

- Retain all Llama body layers. BF16 feature extraction under inference mode;
  no body gradients or optimizer. The body runs once per source. Frozen body
  features are cached, then the body is unloaded before head training.
- Encode the same chat-formatted summarization request for feature extraction
  and the full-parent baseline. No source truncation. Require rendered parent
  prompts <=512 tokens and existing native-compatible source/target lengths.
- Use the existing source-conversation-disjoint native train/test partition.
  Select 2,048 eligible training pairs by SHA256(`frozen-body-train:` + ID).
  Select 64 eligible test pairs by SHA256(`frozen-body-test:` + ID). All test
  data may have been used for previous evaluation; no hyperparameter selection
  on these new test outcomes. The codec/generator have previous training history.
- Freeze the shared-memory-adapted codec completely. Initialize both generator
  arms from the shared-memory-adapted NCM generator. Hybrid conditioning is
  LayerNorm(3072) + Linear(3072,768), newly initialized, replacing native prompt
  embeddings. The native control retains its existing GPT-2 prompt embeddings.
- Train each arm for 512 updates, batch 8, AdamW lr 1e-4, weight decay .01,
  warmup 32 then cosine to .1 of peak; clip gradient norm 1. Seed 42 for model
  and dropout; separate minibatch seed 2042. Identical minibatches and target
  codes in both arms. FP32 trainable weights/optimizer, BF16 compute. No noise
  or other new distillation objective in this first attached-head comparison.
- Supervision is cross entropy on the reference summaries' frozen codec codes.
  No held-out checkpoint selection. Save optimizer/RNG continuation and final
  generator/adapter. The codec's 48-position output canvas is unchanged.
- Full-parent comparison: greedy BF16 generation from the identical rendered
  prompt, <=96 new parent tokens; no sampled best-of selection.
- Evaluate all three frozen outputs on exactly the same 64 cases. Primary judge
  `claude_code/claude-sonnet-4-6`, temperature 0, using the unchanged source-grounded
  whole-summary rubric and conservative-failure parser from the completed audit.
  Pass = supported, salient, readable, nonempty, <=128 cl100k_base tokens.
  Reuse hash-verified 48-control primary calibration from that completed audit;
  author-reviewed controls are not independent human gold labels. Four workers,
  <=3 attempts only for transport/schema failures, preserve every raw reply.
- Report count/N and Wilson intervals; 95% requires at least 61/64 passes.
  Accuracy improvement requires more hybrid passes than the matched native
  control. Neither rule establishes production accuracy or statistical power.
- Frozen-body/module feasibility requires nonzero adapter gradients, finite
  training, unchanged codec, save/reload equality and inference using only
  parent features. Replacing compatible native context with an external-context
  wrapper must reproduce native FP32 teacher-forced logits within atol=1e-4,
  rtol=1e-5, and greedy codes exactly. Separate projection GEMM shapes can cause
  harmless floating-point roundoff. Save/reload logits must match exactly.
- CPU contracts, a CUDA lifecycle/memory pilot (16 train, 2 test cases, two
  updates per arm) and single-worker pilot precede the full run. Pilot metrics
  are systems-only. Physical GPU 0 only; no changes to unrelated GPU jobs.
- Invalid hashes, nonfinite numerics, reload/parity failure or resource failure
  stop quality claims. Preserve failures; recovery requires a new dated root.
  No throughput benchmark or speedup claim in this experiment; feature caching
  is a training optimization, not a production reuse assumption.

Artifacts: `data/experiments/ncm_frozen_body/20261022_{pilot,train}_v1`.
Classification distinguishes completed attached-head training, the 95% quality
gate, and improvement over the matched control. A low quality grade is not retried.
