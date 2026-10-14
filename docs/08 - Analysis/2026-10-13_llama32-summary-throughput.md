# Llama 3.2 3B: 2.40x summarization throughput over optimized FP16

Date: 2026-10-13. Hypothesis `llama32-3b-summary-throughput-v1`. Profile: EXPLORATORY.
Classification: **`twofold_fp16_speedup_with_proxy_quality_supported`**; confirmation remains false.

Aggregate: `data/experiments/llama_summary_throughput/20261013_primary/campaign_results.json`, SHA-256 `36b125ba0e59f51e82a595edd111e2d2bea28da9d2268b451cedace99833d77b`.
[Prospective plan](../07%20-%20Status%20Reports/2026-10-13_llama32-summary-throughput-plan.md) · [Frozen selection](../07%20-%20Status%20Reports/2026-10-13_llama32-summary-selection.md).

The selected existing Q4_K_M checkpoint with target-verified prompt lookup (eight-token drafts) achieved **81.77 output tokens/s**, versus **34.13** for optimized FP16 ordinary decoding: **2.396x**. Mean model-request latency fell from **2.458 to 1.057 seconds (2.327x)**. Against the already quantized Q4 ordinary decoder, the gain is **1.398x**, not twofold. All arms run the complete 28-layer Llama 3.2 3B Instruct model on the same RTX 3060, physical GPU 0.

## Matched measurements

| Configuration | Output tokens/s, including prefill | Mean request s | Decode steps/s, including EOS | Mean output tokens | Reviewed requests |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 ordinary baseline | 34.13 | 2.458 | 45.15 | 83.90 | 20/20 |
| FP16 + prompt lookup | 61.53 | 1.364 | 109.94 | 83.90 | 20/20 |
| Existing Q4 ordinary control | 58.51 | 1.467 | 101.66 | 85.85 | 20/20 |
| Existing Q4 + prompt lookup (selected) | 81.77 | 1.057 | 199.28 | 86.40 | 20/20 |

Primary throughput is summed generated non-EOS tokens divided by summed synchronized model-request wall time. It includes cache clearing, prefill, CPU token selection and generation. It excludes model loading, input tokenization, text detokenization, storage and application/network overhead. These are warm model-inference measurements, not full application latency. Slight output-length differences contribute to tokens/s; the independent latency improvement prevents verbosity alone from passing the gate.

| Input family | FP16 tokens/s | Selected tokens/s | Throughput ratio | FP16 request s | Selected request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw, approximately 500 prompt tokens | 40.28 | 127.88 | 3.175x | 0.655 | 0.244 |
| Raw, approximately 2,000 prompt tokens | 27.99 | 49.74 | 1.777x | 1.086 | 0.611 |
| Raw, eight fragments / approximately 6,300 tokens | 34.56 | 85.46 | 2.473x | 7.362 | 3.014 |
| Hierarchy proxy, approximately 1,000 tokens | 33.40 | 73.97 | 2.215x | 0.731 | 0.357 |

**Twofold is an aggregate result, not a guarantee at every input size.** The 2,000-token family reaches 1.777x. Individual requests improve in median latency by 1.754–3.121x. The synthetic workload gives each input family equal request count; production weighting is unknown. Long-batch Q4 prefill/first-token time is 1.683 seconds, slightly slower than FP16's 1.631 seconds, and remains over half the selected 3.014-second request. Most gains come from generation.

## Technique and quality boundary

The native runner uses the pinned llama.cpp runtime, full GPU residency, FlashAttention, F16 KV state, 8,192-token contexts and 512-token prefill chunks. Prompt lookup matches recent generated suffixes against the current request's prompt only. The target model selects every emitted token, verifies proposed spans in a batch and removes rejected KV tails. No draft network, training, MLP removal, layer reduction, cross-request prefix reuse, source truncation or reduced generation cap is used. Some tokens differ because quantization and batched arithmetic change logits; bitwise equivalence is not claimed. The selected arm accepted 5,610 of 12,800 proposed tokens (43.83%).

References: [Hugging Face assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding), [llama.cpp speculative examples](https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative), and [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md). Runtime commit: `23b0202a189c44a54625aadcb37a946dd1d6278d`.

Twenty held-out synthetic requests from five new seeds cover 55 source fragments. Every observed output variant passed assistant review for final owner/budget/deadline, project and fragment attribution, omitted/extra summaries, obsolete proposals and invented claims. All four arms passed 20/20 requests, with no observed quality regression. Across all 400 timed requests there were 27 distinct outputs; repeated timings are not independent quality samples. The longest individual summary, including its label, was 33 tokenizer tokens, below 128. All normal requests stopped at EOS. `reviews.json` preserves explicit dispositions and source quotations; `review_inventory.json` preserves the complete source/output pairs. Simple string-presence diagnostics were not treated as quality judgments.

**This does not establish 95% production accuracy.** Sources are narrow, repetitive final-decision dialogues that favor copying and prompt lookup. Evaluation was performed by the coding assistant with access to sources, not blinded human reviewers. The production hierarchy input distribution, broader abstractive summarization, phone/NPU performance and application latency remain unmeasured. Text below 192 tokens bypasses summarization and was excluded from model-speed measurements. Raw caps remain 768 generated tokens and hierarchy caps 512; source fragments remain at most 2,048 tokens and raw prompts at most 6,300.

## Storage, capacity and integrity

| Artifact | GGUF bytes | Runtime weight bytes | Sampled resident GPU increment |
| --- | ---: | ---: | ---: |
| FP16 | 6,433,688,032 | 6,425,850,112 | 7,671,382,016 |
| Existing Q4_K_M | 2,019,377,696 | 2,011,539,712 | 3,256,877,056 |

Q4 uses about 68.6% less weight-file space and 57.5% less sampled GPU residency. Residency includes allocated context/workspace and is not a transient peak measurement. Model loading took 1.69–1.75 seconds for FP16 and 0.875–0.890 seconds for Q4 after model hashing; these are not cold-filesystem startup measurements.

FP16 was converted from the pinned original checkpoint; the selected Q4 is the pre-existing vendor artifact. Their file hashes are respectively `4924af1a90334aab0922e60d6285c006485b32809a50f032a9187b8012622174` and `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`. The new naive local Q4 conversion is a different artifact and was rejected after two development batches repeated and mixed summaries. This study does not isolate the quantization detail responsible for that quality difference.

Primary order was FP16 A, Q4 A, Q4 B, FP16 B, with two repeats in A and three in B; strategies alternate within each process. All 400 planned cells completed. The selected prompt, model and draft length were frozen before any held-out outcomes. Source/binary/shared-library snapshots, physical model files, raw hashes, complete grids, timing/token invariants and review coverage were audited. Only GPU 0 was used; approximately 291 MiB of unrelated resident GPU allocation remained. There was no concurrent experiment of ours on GPU 0, but transient external kernel activity was not continuously monitored.

Separate capacity checks ran both models and both decoding strategies at all 768 raw / 512 hierarchy steps with EOS disabled, including a 6,295-token raw prompt. They test capacity through 7,063 total positions, not the last five positions of the theoretical 7,068-token raw maximum. Their artificial continuations and timing are excluded from quality and primary speed. CPU audits overlapped these capacity checks, not primary timing. `capacity_audit.json` records all eight passing cells.

Earlier failures remain preserved: missing SentencePiece on the first FP16 conversion; noisy pilot logging corrected before primary timing; v1 prompts omitting final decisions; v2 prompts generating extra summaries; the rejected overlength draft of v3 and its premature runner launch; and the local Q4 quality failures. The final v3 wrapper states output cardinality and is identical for all arms. Supporting run integrity is separately recorded; an integrity pass does not turn a failed summary into a quality pass.

## Reproduction and validation

Sources: `experiments/structure_net/llama_summary_throughput.{py,cpp}`, `prepare_llama_summary_prompt_v3.py`, and `campaign_llama_summary_throughput.py`. Audit and registry entry points are `experiments/neural_architecture_lab/{analyze,finalize,register}_llama_summary_throughput.py`. Model files, frozen native libraries and raw artifacts remain in the data store.

Use the experiment Python environment with `PYTHONPATH=.:src OMP_NUM_THREADS=2`. Prepare development/held-out fixtures with the producer's `fixtures` command, apply shared prompt v3, then run the campaign against a **new** output root. Do not overwrite the completed primary root. The campaign fixes physical GPU 0 before initialization and serializes model loads. Reproduction uses the pinned native binary or rebuilds it from the saved C++ source against the pinned headers, llama/ggml libraries and CUDA runtime.

```bash
env PYTHONPATH=.:src OMP_NUM_THREADS=2 data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.campaign_llama_summary_throughput \
  --fixtures data/experiments/llama_summary_throughput/20261013_fixtures_heldout_prompt_v3.json \
  --output data/experiments/llama_summary_throughput/NEW_PRIMARY_ROOT
```

Validation: **9 focused tests passed**, including native lookup contracts, immutable fixtures, source-review completeness/tampering, quality-regression veto and speed/latency gates. Real CUDA lifecycle, full-context and capacity probes passed. The typed completed record retains `confirmed=false` and `application_accuracy_tested=false`; exact and semantic registry readback are required before closing the study.
