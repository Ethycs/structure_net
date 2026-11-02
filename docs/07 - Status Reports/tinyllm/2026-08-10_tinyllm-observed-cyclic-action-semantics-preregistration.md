# TinyLLM observed cyclic action-semantics preregistration

**Status:** PREREGISTERED — candidate action outcomes not inspected  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, staged frozen-checkpoint
causal follow-up  
**Hypothesis:** `tinyllm-observed-cyclic-action-semantics-front-v1`

## Prior evidence and new question

The locked observed cyclic-deck campaign established that continuous rotation
of a complete decoded planar observation gives mature `C2` and `C3` quotient
sufficiency in `5/5` d6 checkpoints, but reproduces the separately quantized
generator-defined first-preserved cut in `4/5` `C2` and only `2/5` `C3`
checkpoints. Those outcomes are prior evidence and are not fresh endpoints in
this study.

The two orbit constructors differ in two ways:

1. the observed action rotates the realized sensor noise with the signal,
   whereas the generator keeps the same sensor-frame noise across sheets; and
2. the generator quantizes every sheet separately, whereas the observed action
   operates continuously after decoding.

The new question is:

> Does observable sensor-frame residual transport, rather than re-quantization
> alone, explain the `C3` causal-front displacement?

The directional prediction is that a no-fit action which rotates the coherent
calibrated signal while holding the observation-derived residual fixed will
substantially approach the generator orbit and restore `C3` front replication.

## Frozen sources

Reuse without training or fitting:

- the ten d6 analytic-carrier `k=2,3` checkpoints from
  `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered`;
- the generator-defined reference fronts from
  `data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered`;
  and
- the continuous rotate-all observed-action results from
  `data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_d6_preregistered`.

Validate the three campaign hashes, schemas, configurations, result manifests
where present, per-checkpoint digests, task configuration, and terminal states
before inspecting a candidate action outcome. Conditions are `k=2,3`; seeds
are `7,17,29,41,53`; the frozen checkpoint is the replication unit.

## Observable coherent-signal decomposition

For decoded planar history `x_t`, observed orientation axis `q`, amplitude `a`,
offset `o`, drift `d`, normalized time `t`, and observed signed angular speed
`omega`, form the calibrated complex history

```text
z_t = <q, (x_t-o-dt)/a> + i cross(q, (x_t-o-dt)/a).
```

Estimate current phase from all sensor steps by deterministic demodulation:

```text
c_hat = normalize(sum_t z_t exp(-i omega t)).
```

Construct the coherent planar signal `s_hat_t` by remodulating `c_hat` at the
observed speed and orientation, and define the observed sensor-frame residual

```text
epsilon_hat_t = x_t - o - dt - s_hat_t.
```

For declared cyclic element `alpha`, the residual-fixed action is

```text
g_alpha(x)_t = o + dt + R(alpha)s_hat_t + epsilon_hat_t.
```

The third sensor channel and calibration packet are unchanged. The constructor
may read only the decoded observation, calibration packet, fixed time grid, and
declared group element. Latent/current/future phase, target posterior/bin,
branch, quotient phase, fiber ID, noise realization, nuisance dictionary, and
non-anchor generator rows are forbidden.

## Stage A: input-only action-semantics decomposition

Before loading a model for causal evaluation, compare these fixed constructors
on the existing 256-anchor composition and extrapolation cohorts:

| Constructor | Signal transform | Residual/noise transform | Quantization |
| --- | --- | --- | --- |
| rotate-all continuous | rotate | rotate | none |
| rotate-all requantized | rotate | rotate | serialize and decode each sheet |
| residual-fixed continuous | rotate estimated coherent signal | hold observed residual fixed | none |
| residual-fixed requantized | rotate estimated coherent signal | hold observed residual fixed | serialize and decode each sheet |
| oracle residual-fixed controls | rotate latent coherent signal | hold anchor residual fixed | continuous and requantized |

Oracle controls are attribution-only and can never be selected for the causal
campaign. Requantization uses the existing fixed value serializer and decoder;
it does not retrain or change a model.

For each observable constructor, report:

- relative planar-sensor RMS to the locked separately quantized generator
  orbit (the third channel is outside the analytic carrier action);
- degree-`k` analytic-character angular error against the source anchor;
- direct `2 pi` closure error;
- composition error `g_alpha(g_beta x)` versus `g_(alpha+beta)x` for the
  generator angle;
- corrected planar norm change, support maximum, and finite checks; and
- distance reduction separately by `k` and held-out shift.

### Stage-A selection and stop rule

Select at most one residual-fixed constructor: the one with the smallest pooled
median generator-orbit relative RMS. It is eligible only if, for every `k` and
shift across the five checkpoint cohorts:

1. its median planar relative RMS is at least **50% lower** than rotate-all
   continuous;
2. median degree-`k` character angular error is at most `0.05` radians and the
   95th percentile is at most `0.20` radians;
3. direct closure maximum error is at most `2e-6` for the continuous candidate
   or one decoded quantization step (`0.13`) for the requantized candidate;
4. group-composition relative RMS is at most `0.02` continuous or `0.05`
   requantized;
5. corrected-norm relative error is at most `0.05` at the 95th percentile;
6. planar support is at most `2.0`; and
7. all construction-input and finite-data contracts pass.

Requantization alone supports the alternative explanation only if it achieves
the same 50% distance reduction in every `k` and shift. If no residual-fixed
observable constructor is eligible, stop without model evaluation and record
`no_observable_action_semantics_candidate`; oracle controls may explain the
failure but cannot override it.

Selection reads no TinyLLM activation or output. It is deterministic from the
registered input metrics and cannot be changed after a causal outcome.

## Stage B: frozen causal-front replication

Stage B is authorized only for the Stage-A-selected observable action. Use the
same task/map cohorts, nine cuts, continuation, classification thresholds, and
per-seed reference fronts as the locked observed cyclic-deck campaign:

- analytic carrier;
- block-0 pre-attention, post-attention, and post-MLP;
- block-1 and block-2 post-attention and post-MLP; and
- full depth.

At every cut, average the complete action-generated `Ck` activation orbit and
patch it into the frozen continuation. The matched semantic control uses the
same selected constructor with an added half-task-turn `pi/k`; it must be
evaluated both for its own shifted target and the original source target.

### Primary causal gates

Counts require the same seed to pass composition and extrapolation jointly.
For each `k`:

1. full-depth selected twirling is preserved in at least `4/5` seeds;
2. carrier and block-0 pre-attention twirling are destroyed in at least `4/5`;
3. the first-preserved cut matches the locked generator reference within one
   cut in at least `4/5` seeds;
4. the half-turn control is preserved for its shifted target in at least `4/5`;
5. no more than `1/5` half-turn controls preserve the source target; and
6. replay, state, finite, source, action, and exact-resume integrity gates pass.

The action-semantics hypothesis is supported only if all six gates pass for
both `C2` and `C3`, and `C3` front replication improves from the locked `2/5`
rotate-all result to at least `4/5`. The existing rotate-all result is a fixed
baseline, not rerun or retuned.

## Secondary measurements

Secondary measurements cannot rescue Stage A or B:

- per-cut posterior JS and action-state relative RMS;
- front movement direction and distance per seed/shift;
- continuous-versus-requantized differences within each residual convention;
- observed-phase versus oracle-phase distance decomposition;
- full-depth task gains; and
- analysis time and peak CUDA allocation.

## Outcome meanings

| Outcome | Interpretation | Decision |
| --- | --- | --- |
| residual-fixed passes Stage A and restores both fronts | sensor-frame residual transport explains the earlier front mismatch under an observable action | close action-semantics branch; no retraining |
| residual-fixed passes Stage A but `C3` front remains below 4/5 | input orbit similarity is insufficient to determine the nonlinear causal front | retain mature closure; reject residual transport as the missing mechanism |
| requantization alone clears Stage A | discretization, not residual transport, explains most input-orbit discrepancy | test only the eligible requantized constructor under Stage B |
| no observable candidate clears Stage A | generator convention cannot be reproduced by the declared one-observation action family | stop without model evaluation |
| oracle passes while observable variants fail | the missing variable is clean signal/noise separation unavailable from one observation | do not promote oracle behavior or train from this campaign |
| validity/control fails | implementation or semantic comparison is invalid | quarantine affected endpoints |

No outcome licenses representation penalties, group-loss tuning, probe sweeps,
or TinyLLM retraining.

## Execution and artifacts

Run focused CPU contracts and an input-only Stage-A shakedown first. If Stage B
is selected, run one systems-only CUDA checkpoint before the ten-cell primary
campaign. Store append-only artifacts under:

```text
data/experiments/tinyllm_observed_cyclic_action_semantics/
  20260810_d6_preregistered/
```

Required artifacts include the Stage-A comparison and selection record, every
per-checkpoint result if Stage B runs, diagnostic NPZ files, aggregate strict
JSON, source/implementation/configuration hashes, result manifest, exact-resume
tree digest, measured report, and read-back-verified meta-hypothesis record.
Snapshot the complete data root in DVC and commit remote objects to lakeFS.

## Boundaries

This study tests one deterministic coherent-signal estimator, optional fixed
requantization, synthetic calibrated `C2`/`C3` tasks, the retained d6 checkpoint
population, and the existing N3 composition/extrapolation families. It does not
test learned denoising, unknown calibration, biased or anisotropic noise,
non-cyclic groups, token-space actions, real sensors, natural language, or an
architecture distribution.
