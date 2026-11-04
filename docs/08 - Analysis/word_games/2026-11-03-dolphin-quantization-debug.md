# Dolphin Q4 collapse: diagnosis and mixed-precision repair

Date: 2026-11-03, America/Los_Angeles. Experiment:
`word-games-dolphin-artifact-debug-v1`. Classification: exploratory diagnosis and
deployment validation, not a general accuracy benchmark.

The generation collapse is localized to the low-precision weights **outside the
routed expert bank**. Keeping the original Q4_K_M expert tensors and copying the
remaining tensors from the same-source Q8_0 GGUF recovers coherent responses.
The reciprocal swap—Q8 experts with the original low-precision remainder—still
fails. The repair is installed as the existing `dolphin-x1-trinity-nano` Triton
profile. No training or requantization was performed.

## Controlled results

Six GPU configurations used the same RTX 2060 SUPER, pinned b11960 native runtime,
chat template, system prompt, Q8_0 KV, Flash Attention, 2,048-token context,
256-token batch/microbatch, temperature 0, seed 73, and 64-token output cap.
The twelve fixed prompts cover arithmetic, simple facts, copying, letter order,
extraction, and constrained JSON. Two additional CPU configurations used three
of those prompts, 32 output tokens, FP16 KV and Flash Attention disabled.

| GPU weights | File GB, decimal | Requested content recovered / 12 | Requested output format / 12 |
| --- | ---: | ---: | ---: |
| Original community Q4_K_M | 3.758 | 0 | 0 |
| Community imatrix Q4_K_M | 3.758 | 0 | 0 |
| Previously selected publisher Q5_K_M | 4.385 | 9 | 1 |
| Same-community-source Q8_0 | 6.531 | 12 | 1 |
| **Original Q4_K_M experts + Q8 remainder** | **4.038** | **12** | **1** |
| Q8 experts + original Q4_K_M remainder | 6.251 | 0 | 0 |

Content recovery was reviewed directly from the saved answers by the assistant;
there was no external judge model. It disregards extra prose and spacing and is
not a score of complete response quality. For example, both Q8 variants explicitly
state the reverse ordering `c, b, a` but render it as `cb a`. This receives content
credit and fails format. The initial conservative review scored this as a failure
(11/12); both that review and the explicit scoring clarification are retained.
The automated literal-answer check also reports 11/12 for these variants.

Serious limitations remain: Q8 and the repair correctly answer **6 apples** but
introduce an incorrect explanation about adding pears. The repair copies `ZX47`
but then invents a possible interpretation as coordinates. Q5 instead gives
**8 apples**, incorrectly inserts hyphens in the copied code, and fails reversal.
Thus **12/12 content recovery does not mean 100% accuracy or reliable reasoning**.
These prompts were diagnostic probes, not a held-out application benchmark.

The original Q4 also fails all three CPU probes with garbled repetition. Its
failure therefore is not exclusive to CUDA, Flash Attention, or Q8 KV. The CPU
Q5 control recovers the facts but degenerates into repetition on the Paris answer;
that control limitation was recorded and the initial CPU comparison plan amended.

## Evidence for the cause

Q4 and Q8 have identical architecture/tokenizer metadata, all 1,057 tensor names
and shapes agree, and **all 445 FP32 tensors match byte for byte**. Their metadata
differences are only quantization type and quantization timestamp. Q4 versus the
publisher Q5 also has 445/445 identical FP32 tensors and identical sampled Q6
values. Descriptive Base/Preview metadata differences alone were consequently
not evidence of substantially different checkpoints.

The successful and unsuccessful mixtures were formed by tensor copies. All 1,057
output tensor hashes were checked on readback for each file. This is a controlled
group intervention: raising the routed expert bank alone is insufficient; raising
the other quantized weights is sufficient to remove the observed collapse on
these probes. The latter group contains embeddings, attention, the leading dense
MLPs, shared experts and output weights. FP32 normalization and routing weights
are unchanged. We have **not** isolated an individual matrix or established the
numerical mechanism of its sensitivity.

The original Q4_K_M routed bank itself mixes Q4_K and Q6_K. The repaired artifact
contains 136 Q4_K, 26 Q6_K, 450 Q8_0, and 445 FP32 tensors. It is 7.5% larger than
the failed Q4 file, **38.2% smaller than Q8**, and **7.9% smaller than the prior Q5**.
No broad quality equivalence to Q8, immunity to other quantization/runtime issues,
or throughput improvement is established by these experiments.

## Deployment and reproduction

Selected weights:
`/data/models/Dolphin-X1-Trinity-Nano-GGUF/Dolphin-X1-Trinity-Nano.Q4_K_M-experts-Q8_0-rest.gguf`

SHA-256: `c40c5976e13a101dc24694c4c36c744bd45e754c2e78b816e9292518df799b30`.
Size: 4,038,439,776 bytes. The pinned inputs and deterministic copy rule are in
[the recipe](../../../deployments/word_games_triton/fleet/dolphin-mixed-precision.json).
Original checkpoints and diagnostic mixtures remain available for comparison.

At the normal **8,192-token deployment context**, live checks passed chat history,
JSON, streaming with usage, direct Triton JSON inference, model identity, and
unloading. The 2060 SUPER reported 4,186 MiB used including existing display
allocation, with 3,601 MiB free. All 57 transformer/output layers were offloaded;
the native log also records a 207.72 MiB host model buffer. Unloading completed
in 1.31 seconds. There is no representative long-context quality measurement.

The route keeps its existing alias, GPU placement, native optimizations and KV
settings. The live transport checks explicitly allow explanatory text while
retaining separate exact-format results. Fifty-two focused tests passed, including
rejection of stale integrity receipts and wrong derived weights. All 636 installed
package files were verified after reinstalling the catalog packages.
The router and Empero were restored to their pre-experiment resident state;
`restoration.json` records the final status. Dolphin is loaded on demand.

From the repository root, use Pixi:

```bash
pixi run python -m experiments.word_games_triton.prepare_dolphin \
  --receipt data/experiments/word_games/<fresh-run>/dolphin-weights.json
pixi run word-games-load dolphin-x1-trinity-nano
```

The preparation command downloads missing revision-pinned inputs, checks available
space, verifies checksums, copies the selected tensors, verifies readback, and
requires the expected final SHA-256. Existing Triton load/unload commands continue
to work. LiteLLM registration remains a separate infrastructure handoff to 201-1002;
these native tests do not establish that the new gateway alias is registered.

Raw evidence is under
[`data/experiments/word_games/20261103_dolphin_debug_v1`](../../../data/experiments/word_games/20261103_dolphin_debug_v1).
It includes the plan and dated amendments, producer hashes, metadata snapshots,
download receipts, tensor comparisons/copy manifests, every request and response,
native logs, initial and clarified content reviews, and live deployment receipts.
The reusable harness is
[`dolphin_debug.py`](../../../experiments/word_games_triton/dolphin_debug.py);
the copy implementation is
[`mix_dolphin_tensors.py`](../../../experiments/word_games_triton/mix_dolphin_tensors.py).

The model sources are the publisher's
[Dolphin X1 Trinity Nano](https://huggingface.co/dphn/Dolphin-X1-Trinity-Nano),
the pinned [static GGUF repository](https://huggingface.co/mradermacher/Dolphin-X1-Trinity-Nano-GGUF/tree/58536b87c91b4cec596a97e9e7ae0b06d8ba1170),
and the [imatrix repository](https://huggingface.co/mradermacher/Dolphin-X1-Trinity-Nano-i1-GGUF/tree/f4c6644786a0a3fb760a27120bd1b1c0826e5215).
The latter explicitly identifies its files as weighted/imatrix quantizations;
its failure here does not establish that calibration cannot help other recipes.
