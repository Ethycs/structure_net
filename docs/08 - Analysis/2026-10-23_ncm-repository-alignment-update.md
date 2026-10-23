# NCM repository alignment and corrected decoder audit

Date: 2026-10-23. This supplements the
[pinned repository comparison](2026-10-23_ncm-reference-repository-comparison.md)
with completed follow-up results and a correction to an auxiliary source note.
The earlier raw artifacts remain unchanged.

The runnable NCM reference is
[HazyResearch/ncm](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203).
Our saved 32-output comparison matched the author inference path exactly, with
ROUGE-1/2/L of 34.6064/12.9796/27.8921. This small inference control supports
using the release as a reference; it is not a reproduction of the full benchmark,
its missing trainer, or the application's 95% summary-quality target.

## What the implementation requires

| Mechanism | Official released NCM | Failed native hybrid | Action for the attached head |
| --- | --- | --- | --- |
| Representation | Learned encoder, residual VQ hierarchy, learned text decoder | Clustering over frozen native embeddings | Retain and qualify a learned codec before predicting its codes |
| Finest level | Full 48-position target, with nine levels and 4,096 entries per level | Coarse hierarchy ends at 64 positions for a 128-word stage | Include a code level at the full represented target length |
| Conditioning | Reconstructed earlier-level latents; block attention within each level | Causal native-word refinement from predecessor estimates | Use the actual hierarchy input construction and level masks |
| Word output | Dedicated decoder reads unshifted current-position reconstructed latents | Original causal LM head needs a correct previous-word history | Use a dedicated parallel decoder; preserve its actual attention mechanism |
| Parent attachment | Released standalone conditional generator | New adapters through the original Llama body | Provide per-token frozen-parent source features to the new module; test this adaptation separately |
| Training | Release contains inference modules/checkpoints, not a trainer | Several codec, loss and trajectory changes at once | Isolate codec reconstruction, conditional code prediction and source-only task quality |

The new blockwise interface exposes four independent released 48-position
codec blocks as one level-major hierarchy. CPU tests check exact code and
decoder parity, earlier-level-only conditioning, no cross-block latent mixing,
gradient routing through the frozen decoder, and the actual released generator's
forward/backward and greedy generation. These are interface tests, not evidence
that a trained conditional summarizer works.

## Correction: decoder attention is bidirectional

The saved auxiliary `decoder_attention_supplement.json` incorrectly inferred
decoder causality from `models/common.py`. The decoder actually instantiates
`Block` and `SelfAttention` defined in
[models/vae.py](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vae.py).
Its SDPA call has no causal flag and no mask. The pinned official file and the
local runtime file have identical SHA256
`7d393b56813342278def3b053e83ec4ff4e56a492e4ddf2a2441e1c9f78d4317`.

The new `20261023_decoder_attention_resolution_v1` artifact records a CPU probe
of the actual released 60.63M-parameter codec. Both observed decoder attention
calls had `is_causal=False` and no mask. Perturbing only the last latent changed
the first-position word logits by a maximum absolute 0.493434906. All codec
state tensors remained unchanged. The earlier causal interpretation is
retracted; it did not change the modules used by any completed numerical run.
The dedicated decoder is both bidirectional and conditioned on current-position
latents. Either difference matters when comparing it with a native causal head.

## What the training repositories supply

[Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM/tree/22775363f8563c63620e71f7a204e90a51d6a379)
has Llama trajectory collection, paired training and inference code.
[JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing/tree/028dbd6283fdc6fa356f0b109ce04c5ca538e49f)
provides a newer causal parallel-decoding pipeline, including trajectory packing,
padding/EOS handling and training. The important training alignment is the exact
model's own converged outputs and intermediate states, with a strong clean
objective retained alongside the consistency objective. External summary labels
and arbitrary noisy states are not automatically that model's fixed points.

Those repositories are alternatives for accelerating the existing causal
summarizer. They are not NCM trainers, and their reported results do not validate
our frozen-body 3B summarization setup. Their full training was not reproduced
locally. Keep that branch separate from the requested NCM output replacement.

## Updated local evidence

The isolated clean native teacher completed 512 updates and scored **56/64
(87.5%)** complete-summary passes, below 61/64. This supersedes the older
comparison's description of that run as live. It is not an NCM model. The prior
converged NCM hybrid remains **41/64**; its original fixed nine-pass version
remains **0/64**. Neither final cohort has been opened.

The new learned-codec transfer completed 2,048 updates. Its first version fell
from released **95.179%** to **67.167%** oracle token recovery. A matched
sample-scaled EMA control recovered **94.262%**, with all sampled windows and
learning rates identical. It remains below the released codec and fails the
99.5% representation gate. Those percentages are token reconstruction, not
complete-summary accuracy. [Matched EMA report](2026-10-23_ncm-sample-scaled-codec-ema.md)

The prospective semantic audit compares uncompressed targets and both codec
reconstructions against their complete public sources. It is required to
distinguish harmless wording changes from lost facts; target-derived codes
still make it an oracle component control. Its eventual results cannot establish
conditional generation or retrospectively pass either failed reconstruction
gate. [Semantic audit protocol](../07%20-%20Status%20Reports/2026-10-23_ncm-oracle-codec-semantic-plan.md)

The next NCM training decision should follow that component evidence. Retain
the parent's original weights, train the new head's own modules, and measure
complete source-only summaries across the application range before comparing
prefill, decode latency and tokens per second. No local NCM quality or speedup
claim is established by finding runnable upstream inference code.
