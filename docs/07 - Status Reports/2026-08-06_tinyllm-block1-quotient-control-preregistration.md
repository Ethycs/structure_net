# TinyLLM block-1 quotient-control preregistration

**Status:** PREREGISTERED DESIGN with post-launch integrity amendment

**Hypothesis:** `tinyllm-block1-horizontal-vertical-control-v1`

## Question

Can explicit control of semantic-base fidelity and nuisance-fiber contraction
at the post-MLP residual of block 1 turn the support-relative N3 cosine code
into a stable internal quotient?

The comparison has two conditions over seeds 7, 17, 29, 41, and 53:

1. the retained ordinary N3 checkpoint;
2. a matched block-1 quotient-controlled N3 model.

The ordinary checkpoints are reused byte-for-byte. The controlled model uses
the same d8 architecture, initialization seed, 4,096 training examples, paired
minibatch indices, 600 optimizer steps, AdamW hyperparameters, and model-only
gradient clipping. A zero-weight regression test reproduces the ordinary final
state exactly.

## Frozen intervention

Let `r1` be the final-token post-MLP residual of block 1, `u = cos(phi)`, and
`b` the phase branch. Before the temporary heads, `r1` receives parameter-free
per-example layer normalization.

The base head is linear and minimizes:

`L_base = MSE(a(r1), u)`.

The branch adversary is a width-128 two-hidden-layer GELU network conditioned
on the true cosine. Gradient reversal is an identity on the forward pass and
negates the residual-side gradient. The implemented saddle objective is:

`L = L_task + 0.2 L_base + 0.2 CE(D(GRL(r1), u), b)`.

Thus the adversary minimizes positive branch cross-entropy while the
transformer maximizes it. The base sign in the original sketch was corrected:
subtracting MSE under minimization would destroy horizontal fidelity. The
transformer and temporary heads use separate AdamW optimizers; auxiliary
gradients cannot alter the transformer's clipping norm. There is one adversary
update per transformer update and no warmup.

The weights `0.2/0.2` are frozen before confirmatory seeds and match the scale
of the preceding quotient intervention. Composition and extrapolation data are
not used for training, early stopping, or coefficient selection.

## Confirmatory endpoint

Fresh frozen nonlinear probes use the existing N3 interpolation
train/validation protocol. The temporary training heads never define success.
At query, post-attention, post-MLP, and full-depth cuts, the study reports:

- cosine Pearson correlation;
- cosine-conditioned balanced branch accuracy;
- conditional log-loss gain, defined as cosine-only CE minus residual-plus-
  cosine CE.

For a seed, the representation gate is the joint conjunction at both post-MLP
and full depth on both composition and extrapolation:

`correlation >= 0.90`, `branch accuracy <= 0.55`, and `log-loss gain <= 0.02`.

Full-depth exact-bin task accuracy must also remain within three percentage
points of its paired ordinary checkpoint on in-distribution, composition, and
extrapolation examples. The same seed must pass all seven requirements. The
campaign succeeds only if at least four of the same five seeds pass jointly.

In-distribution representation metrics and query/attention cuts are
mechanistic measurements, not additional representation gates.

## Local finite-perturbation diagnostic

At every recorded cut, the study uses interior cosine anchors
`(-0.7, -0.35, 0, 0.35, 0.7)`, `delta = 0.05`, and 16 nuisance replicates. The
semantic stencil holds branch, direction, nuisance parameters, and noise draw
fixed. The opposite-branch chord also uses identical direction and nuisance
draws at its two ends. This avoids confounding branch separation with nuisance
variation.

The diagnostic reports numerator and denominator separately and their median
ratio after diagonal nuisance-residual whitening:

`Q_local = ||J_base|| / (||J_branch|| + 1e-8)`.

It also records token Hamming fractions because the generator is quantized.
This is a finite-perturbation diagnostic, not an analytic Jacobian and not a
success gate.

## Decision table

- Joint success: explicit horizontal/vertical control created a stable tested
  quotient.
- Composition only: quotient formation remains support-relative; test an
  equivariant sensor encoder next.
- Branch erased but cosine lost: compression, not a quotient.
- Cosine preserved but branch remains: the adversary or intervention cut is
  insufficient.
- No joint improvement: stop adding representation losses and test an
  invariant/equivariant front end.

## Boundaries

The N3 family is a broad axis-coverage distribution with a held-out
amplitude/orientation composition, not a literal nested-support theorem. Frozen
probes bound tested decodability rather than conditional mutual information.
The primary probe protocol is held fixed for comparability; a stronger
multi-restart probe audit remains a useful secondary robustness check if the
confirmatory gate appears to pass.

## Post-launch integrity amendment

The user-supplied design, thresholds, seeds, and outcome meanings preceded all
training. The implementation fixed weights at `0.2/0.2` and passed its objective
and zero-weight equivalence tests before primary results were inspected.

The first campaign began while source hardening was still in progress. Some
cells had completed, but their outcomes had not been used to change any
scientific choice, when this risk was identified. That run was excluded because
spawned workers could have imported different code revisions. After partial
outcomes were visible, the only experiment-code change was an implementation
SHA-256 admission check; architecture, data, examples, minibatches, steps,
optimizers, weights, probes, thresholds, and seed rules remained unchanged. A
clean full rerun was written to:

`data/experiments/tinyllm_block1_quotient_control/20260806_d8_code_frozen/`.

The exact clean command was:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache pixi run python -m \
  experiments.structure_net.tinyllm_block1_quotient_control \
  --gpus 1 --slots-per-gpu 0 --max-parallel 2 \
  --output data/experiments/tinyllm_block1_quotient_control/20260806_d8_code_frozen
```
