# Frozen-parent source interface for an attached NCM head

Implementation: `experiments/structure_net/ncm_frozen_source_head.py`.
This is a tested interface, not a trained summarizer or a quality result.

```mermaid
flowchart LR
    S[Complete source prompt] --> P[Frozen parent transformer body]
    P --> H[Every source token's hidden state]
    H --> B[New normalization and projection]
    B --> G[New conditional NCM generator]
    G --> C[Coarse-to-fine code levels]
    C --> D[Frozen qualified learned codec decoder]
    D --> O[Parallel summary words]
```

`frozen_parent_features(parent, input_ids, source_mask)` calls the original
body with `use_cache=False` and returns all per-token hidden states, detached.
It does not compute the original vocabulary head. The parent must already be
in eval mode with every parameter frozen. The function preserves the complete
input frame; it does not pool or truncate. Mask-derived positions handle left
padding. All original transformer layers, including their MLPs, still run once
during source encoding; no parent weights train.

`FrozenSourceConceptHead(generator, parent_hidden_size)` adds a new LayerNorm
and a projection from the parent hidden dimension to the codec latent dimension.
The projection starts with normal weights of standard deviation
`1/sqrt(parent_hidden_size)` and zero bias. The caller supplies the generator;
the interface does not reset its weights. Only new-module weights receive
gradients. Source features are detached again at the boundary, and padded
features are masked before normalization/projection.

The supplied generator must use released-style parallel level-zero queries,
RoPE, level embeddings, eager attention and the explicit block-diagonal level
mask. Unsupported context encoders, autoregressive root generation, extra root
prediction and post-upsampling convolutions are rejected. The codec must be
frozen in eval mode. Four independent 48-position codec blocks can use
`BlockwiseCodec`; the generator then has 800 code positions across nine levels,
with a full 192-position finest level. Source width must equal the configured
prefix width. A too-short frame raises an error instead of clipping input.

Training `head(features, mask, target_codes)` conditions each level on the
reconstruction from earlier code levels. Current and future gold codes do not
enter that level's input. `head.eval().generate(features, mask)` constructs
the root queries and predicts all levels greedily from source features alone.
It uses the same eager per-level computation as the reference, with no target
codes, previous target words, autoregressive fallback or external verifier.
Text decoding remains a separate call on the frozen codec. No cross-request
prompt lookup is involved.

Three CPU integration tests use the actual released codec and generator classes
at small dimensions, plus a small Llama body. They establish exact greedy code
parity against the author generator for equivalent source embeddings, full
source-feature shapes, padding isolation, earlier-level-only conditioning,
nonzero new-head gradients, no gradients into source features or parent/codec
weights, unchanged parent weights, source-only generation and capacity rejection.
They do not establish long-context numerics, training convergence, summary
accuracy or throughput. The weight-migration test retains every learned
generator parameter, repeats the learned root queries for independent blocks,
rebuilds only the attention/level geometry buffers, and rejects changed codec
representation semantics.

Before conditional training, the codec still needs qualified semantic fidelity.
`migrate_generator_state` supports the explicit independent-block geometry
change. It requires an exact reference encoder/quantizer representation, retains
every learned generator parameter, repeats root queries per block and rebuilds
the attention/level buffers. The small-class test does not establish a qualified
full-size checkpoint migration or long-context model quality.
Then freeze the exact producer, run a representative CUDA pilot, train the
new modules on source-disjoint labels, and evaluate complete source-only
summaries across the application range. Both final cohorts remain closed
until the development gate is met. Speed measurement follows quality.
