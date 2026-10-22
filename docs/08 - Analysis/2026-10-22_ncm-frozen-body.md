# NCM trained on a frozen Llama body

**Hybrid summary task accuracy: 0/64 = 0.00%.**
The parent body and codec remained unchanged; the attached NCM module trained.

Protocol: [fixed design](../07%20-%20Status%20Reports/2026-10-22_ncm-frozen-body-plan.md).

Date: 2026-10-22. Hypothesis: `ncm-frozen-llama-body-v1`. One seed, exploratory.
Classification: `frozen_body_training_succeeded_quality_below_95`.
Aggregate: `data/experiments/ncm_frozen_body/20261022_train_v1/campaign_results.json`.
SHA256: `b514e093a892cd2dd4cdf46cdd018d900d41371fa5b233f623c6764b98e406b3`.

| Model | Passing complete summaries | LLM-judged task accuracy | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| hybrid | 0/64 | 0.00% | 0.00%–5.66% |
| native | 0/64 | 0.00% | 0.00%–5.66% |
| parent | 59/64 | 92.19% | 82.98%–96.62% |

The hybrid is the retained full Llama-3.2-3B-Instruct body followed by a newly
initialized LayerNorm/linear adapter and the existing NCM generator. Only the
adapter and generator trained. The codec was frozen. The native control used
the same initialized generator/codec, 2,048 eligible training pairs, minibatches,
512 updates and optimizer schedule, with its ordinary prompt representation.
The parent baseline uses greedy generation from the same rendered prompt.
Adapter initialization seed is 42, dropout 52 and minibatches 2042. The dated
[seed clarification](../07%20-%20Status%20Reports/2026-10-22_ncm-frozen-body-seed-clarification.md)
records the already-frozen implementation's derived dropout seed; no run changed.
The new training producer omitted explicit GPT-2 tokenizer hashes. Before
grading, finalization checked every cached file against the immutable prior
training producer and retained `native_tokenizer_readback.json`; no tokenizer,
producer or generated output was changed. This is a late provenance check.

The 64 cases were selected by fixed ID hash from the source-conversation-disjoint
test partition before training. Both source tokenizers must fit their 512-token
limits; no source or target truncation. The codec has a 48-position output canvas.
These are short fragments, not the full application input/output range.

The unchanged source-grounded rubric requires faithfulness, main-point usefulness,
readability, nonempty output and <=128 cl100k_base tokens. Model identities were
hidden and candidate order shuffled. The primary judge's hash-verified calibration
was reused: 24/24 acceptable controls and 24/24 bad controls correctly classified.
Controls are agent-authored/source-reviewed, not independent human gold labels.
All 64 passages are exact spans of the public LongMemEval-S dataset. Its local
file SHA256 matches Hugging Face's published LFS SHA256; the provenance receipt
is retained. No training targets, credentials, private corpus or internal paths
are included in judge payloads. The original attempted grading launch was
rejected before execution; public-data verification preceded the reviewed retry.
Rates are LLM-judged task estimates; Wilson intervals and every raw response,
retry, source citation and per-case decision are retained in the aggregate/data.

The point target requires at least 61/64 passes. Target met: `False`.
Hybrid improves over matched native control: `False`.
No checkpoint selection or additional training used these final grades.

Frozen-body feasibility is separate from the task-quality result. The body was
run in inference mode and its entire state hash matched before/after extraction;
it was unloaded before optimization. The entire codec hash also matched after
both arms. Adapter gradients were nonzero, the external-context wrapper passed
native logits/code parity, minibatches matched and saved weights reloaded with
identical logits. CPU contracts and a separate CUDA lifecycle/memory pilot passed.

Training used cached frozen features to avoid recomputing them on every update.
This does not assume application prefix reuse and does not measure a deployment
speedup. No throughput claim is established. No neighborhood noise was added in
this first attached-head test. Prior saved targets include low-salience snippets;
their quality remains a limitation on this training objective.

## Reproduction

Use the moba environment Python from the repository root, `LD_LIBRARY_PATH=''`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `PYTHONPATH=src:.`, and physical GPU 0:

```text
python -m experiments.structure_net.ncm_frozen_body pilot --output NEW_PILOT_ROOT
python -m experiments.structure_net.ncm_frozen_body train --pilot NEW_PILOT_ROOT --output NEW_TRAIN_ROOT
```

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` before either GPU
command starts. For CPU/network grading, add
`data/experiments/ncm_memory/dependencies` to `PYTHONPATH` and set
`TIKTOKEN_CACHE_DIR=data/experiments/ncm_memory/tokenizer_cache`:

```text
python -m experiments.structure_net.ncm_frozen_body_public_data --root NEW_TRAIN_ROOT
python -m experiments.neural_architecture_lab.finalize_ncm_frozen_body grade --root NEW_TRAIN_ROOT --report NEW_REPORT
python -m experiments.neural_architecture_lab.finalize_ncm_frozen_body register --root NEW_TRAIN_ROOT --report NEW_REPORT
```

## Fresh body-to-head inference check

A separate integration pass loaded the saved hybrid and processed the first
fixed-hash test source through the actual frozen body, adapter, NCM generator
and decoder, without cached features. It completed on GPU 0 with 7,217,828,352
peak allocated bytes. This is a systems check, not a throughput benchmark.

The fresh batch-one pass and the cached-feature evaluation (body batch four)
differed at six of 48 native output token positions. The cause is not established;
different BF16 batch shapes are a candidate. There is no bitwise end-to-end
parity claim. Save/reload logits on the same supplied context and the native
external-context wrapper checks did pass. The 64-case accuracy table above
measures the declared cached-feature evaluation pipeline.

The separate source, output, producing CLI and comparison are retained under
`data/experiments/ncm_frozen_body/20261022_live_inference_v1/`. Its comparison is
`comparison.json`; this single integration case is not additional quality data.
The reusable inference entry point is `experiments/structure_net/infer_ncm_frozen_body.py`.

## Completed checks and result accounting

Twelve focused CPU tests passed, including gradient routing, context cleanup,
source-length filtering, public-data guards, grading boundaries and conservative
registry records. The CUDA pilot passed before the main run. Both 512-update
arms completed, all 64 primary judge packets are valid, and all final raw replies
were reparsed and matched to the exact frozen requests.

Judge packets retried for transport/schema failure: 1. Valid low grades were not retried. Neither NCM arm passed any complete-summary case. The full parent passed 59/64; its positive performance also shows that the rubric accepts useful model-generated summaries. This one-seed experiment establishes the frozen-body training path, not a useful NCM summarizer or an inference speedup.
