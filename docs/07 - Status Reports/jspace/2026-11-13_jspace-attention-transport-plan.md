# JSPACE-ATTENTION-TRANSPORT-01 — prospective compact attention and transport pilot

Status: prospective protocol, frozen with producers before collection. One checkpoint,
exploratory pilot; `confirmed: false` regardless of outcome. Follow the experimental
master guide and NAL-STD-EXPERIMENT-v0. No model weight training or service changes.

## Question and mathematical object

Can a compact country concept in an actual attention write causally change an answer,
and can its observables predict how that intervention travels through consecutive
attention and MLP updates? We study nonzero observable transport and cross-layer graph
maps. Empty within-layer vanishing ideals do not block this experiment. Degree-two
regression is a baseline candidate polynomial action, not a Gröbner certificate.

Use the frozen local Qwen2.5-3B-Instruct checkpoint identified by SHA256 in the parent
active-CAV study; 36 blocks, width 2,048, BF16, eager attention. Capture the output of
`self_attn` after `o_proj`, the residual immediately after attention addition, and the
block output before the next normalization, at every block 11 through 35 (zero based).
Observe real attention/MLP transformations, with all remaining model layers intact.

## Data and compact bank

Fresh JSON assignments map four split-disjoint names to China, France, Germany, Russia.
Ask for the assigned country's capital or nationality adjective. Answers are respectively
Beijing/Paris/Berlin/Moscow and Chinese/French/German/Russian. A paired donor has the
same assignment prefix and asks about a different person. Only token-length/position
alignment and task construction may filter examples before outcomes are observed.
Maximum 192 prompt tokens. Greedy complete-word generation, at most eight tokens.

Seed 541; 384 training, 24 development, 128 evaluation, four separate systems-shakedown
worlds. Named assignment digest cannot repeat. Training/development source and donor
queries both omit China/nationality and Russia/capital. Evaluation has 16 worlds per
country/attribute cell: 96 seen and 32 held-out combinations. This is a holdout for
fitted probes and maps, not unseen pretrained knowledge. Donors are paired observations,
not additional independent worlds. Evaluation is country-balanced; training is not.

Fit a selected J-lens dictionary for eight single-token lexical rows (each country
bare and with a preceding space). Calibration is 64 unique cached WikiText training
paragraphs of at least 80 words, ordered by SHA256(seed:row_id), truncated to 128 model
tokens. No task examples enter dictionary calibration. Target is block 35 pre-final-norm
hidden state; sum country unembedding cotangents over target positions 1..T-2, average
source gradients over positions 1..T-2, average contexts. Two batched cotangents.
This final-block target is an explicit variant; eight selected rows and 64 contexts
are not the full sparse nonnegative J-space construction. Unit-normalize rows for
readouts; use their rank-at-most-eight linear span for compact edits.

The compact edit is at most eight coordinates **per token**, not eight scalars for the
whole prompt. Pooled observables retain two eight-row readouts: question mean excluding
the final token, and the final token. Basis storage and BF16 reconstruction effects
must remain explicit. This study does not compress model weights or benchmark speed.

## Origin selection and controls

Use only 24 development worlds to compare attention layers 11,17,23,24,25,26,27,28,29.
Methods: orthogonally projected donor-minus-source update in the eight-row J span;
and source/target lexical coefficient swap in their four-row span. Every edit covers
all prefill positions, never cached continuation tokens. Controls: dense donor attention
replacement and a signed/permuted version of each J update with identical norm/rank.

Among layers with at least 50% dense target success on jointly clean-correct pairs,
select highest J target count, then J-minus-random count, then earliest layer, then
projected donor before coefficient swap. If none qualifies, retain the best descriptive
origin and still run the trace; do not claim causal transport. Freeze layer/method
before fitting/evaluation. An attention identity patch must preserve exact token IDs
and recorded scores. Attention-write and post-addition VJPs must be identical.
Save/load dictionary must be exact. Unit tests additionally compare actual attention
VJPs to finite differences and verify exact covector pullback through actual `o_proj`.

## Transport fit and evaluation

Capture natural source/donor scores at all 75 boundaries for training/development.
Fit linear country probes for selected attention write and every subsequent residual
boundary. The transport chain starts at the selected **post-attention residual**:
the attention output alone omits the skip contribution. Follow post-addition -> block
output -> next post-addition -> next block output through block 35.

Standardize each node's 16 coordinates by training mean/std (minimum std 1e-4).
Fit linear and degree-two ridge maps between successive nodes using the 16 coordinates
and known binary requested attribute. No future activations or answer/country labels
are map inputs. Training uses 768 paired observations, 384 independent worlds.
Select ridge penalty from 1e-4,.001,.01,.1,1,10 using development MSE; country probes
select accuracy then MSE. Freeze fit and hash before evaluation. CPU fit is an explicit
small coordinate-space operation; model and Jacobians require GPU reservation.

Evaluation: 128 new source/donor worlds; dense, selected J, matched random, donor
orthogonal remainder, and alternate J method. The remainder complements projected
donor, **not** a coefficient swap. Only the selected J arm gets full downstream trace.
Report exact complete-word teacher accuracy, target-answer intervention success,
paired J-minus-random exact test, pure attention country readability, and actual
downstream country readability. Readability is not itself causality.

Initialize separate clean/edited rollouts from their observed starting residual
coordinates, then compose fitted maps with no later actual activations. Score both
natural states and, primarily, predicted edited-minus-clean changes. Compare against
linear maps and carrying the initial standardized change unchanged. Primary destination
is chain index 7: block output at selected layer+3. Evaluate every world, not only
successful interventions, and report the 32 held-out combinations separately.

Normalized change error = L2(predicted change - actual change)/L2(actual change),
aggregated over worlds/coordinates. Nonfinite or absolute standardized value >=1e4
is divergence; failed trajectories must not be dropped or credited. Any failure makes
that aggregate error null/diverged. Zero-reference error is undefined, not zero.

Prospective descriptive gates:

- Source and donor accuracy each >=90%, jointly correct >=80%.
- Dense target success >=50%, selected J >=20%, J-minus-random >=10 percentage points,
  paired exact two-sided p<=.05, all among jointly correct pairs.
- Polynomial transport signal: primary change NRMSE <.75 and at least 10% below both
  linear and carry controls. This is a pilot effect threshold, not an independent
  statistical confirmation. It supports a mechanistic interpretation only if origin
  causality also passes. All results retained regardless of gates.

No exact closure, full J-space, Gröbner-specific benefit, lossless compression, or
general task-quality claim follows. No PCA transport comparator in this initial pilot.

## Execution and lifecycle

Root `data/experiments/jspace_attention_transport/20261113_v1` is append-only. Freeze
model identity, task/calibration digests, sources, tests, this protocol, Pixi lock,
inputs and resource-client code; archive producers. Separate prepare, discovery,
training capture, CPU fit, evaluation, analysis stages. Strict JSON, stage checksums,
failure records, exact artifact readback. Do not edit frozen producers to repair a
failed stage; archive the failed root and create a declared revision instead.

Use root locked Pixi (`-e ncm` on GPU); every GPU stage uses inference-ctl `lease run`,
3060=10240 MiB, at most 60 minutes, with guard before any CUDA query/load and periodic
lease checks. No ambient GPU jobs, unmanaged Python environment, or service restart.
CPU work has CUDA_VISIBLE_DEVICES empty and four BLAS threads. Release every lease.
Register conservative pilot evidence and read it back in a fresh process. Report
failure mechanisms and examples, not only aggregate pass/fail labels.
