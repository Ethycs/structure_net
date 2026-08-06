# TinyLLM block-1 quotient control

**Status:** FAILED PREREGISTERED GATE — modest horizontal improvement, worse
compositional branch leakage

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-block1-horizontal-vertical-control-v1`

**Conformance:** `PREREGISTERED` with the integrity amendment recorded in the
[preregistration](../07%20-%20Status%20Reports/2026-08-06_tinyllm-block1-quotient-control-preregistration.md)

## Verdict

Explicit block-1 horizontal/vertical control did **not** create a stable
nuisance-invariant internal quotient. Zero of five controlled models passed the
representation conjunction and zero passed the full representation-plus-task
gate; four were required.

The intervention preserved compositional cosine and modestly increased mean
extrapolation cosine, from 0.458 to 0.544 at block-1 post-MLP and from 0.452 to
0.556 at full depth. Those values remain far below the preregistered 0.90 gate.
The adversarial half of the intervention also moved in the wrong direction on
composition: mean conditional branch accuracy rose from 55.4% to 68.4% at
post-MLP and from 59.7% to 63.6% at full depth.

This is not merely compression: the linear base head learned its training
coordinate, but the cosine-conditioned online adversary did not eliminate the
branch readable by a fresh frozen probe. The result supports the preregistered
stop condition: do not add another residual representation loss; test a
genuinely invariant or equivariant sensor front end.

## Campaign integrity

The clean campaign completed 10/10 cells with no failures: five byte-for-byte
retained ordinary N3 checkpoints and five newly trained controlled models.
Every clean cell records the same implementation digest
`e9c365d2ad9bc6dd9098fb26cf40bd1e09129eaf65d4b3af2c796cd6ce52dd50`
and a distinct scientific fingerprint. A two-step CPU lifecycle passed, resume
reused 2/2 cells, and a zero-weight regression reproduced the ordinary model's
final state exactly.

An earlier development run is excluded. It overlapped source hardening, so
spawned seeds could have imported different worker revisions. After partial
outcomes existed—but without changing the intervention, data, weights, probes,
or gates—an implementation-digest admission check was added and the entire
campaign was rerun in a fresh directory. The excluded run is retained at
`data/experiments/tinyllm_block1_quotient_control/20260806_d8_preregistered/`.

The clean runner used two free-memory-calibrated slots on PyTorch logical GPU 1
(the RTX 2060 SUPER, physical NVIDIA-SMI index 2). Each controlled worker peaked
at 2.004 GiB allocated and 3.537 GiB reserved.

## Fixed design

- d8 TinyLLM, 50,964,992 inference parameters;
- seeds 7, 17, 29, 41, and 53;
- N3 training support, 4,096 examples, paired batch size 64, 600 updates;
- AdamW at `3e-4`, weight decay `0.01`, model-only gradient clip `1.0`;
- parameter-free normalization followed by a temporary linear cosine head;
- width-128 nonlinear branch adversary conditioned on exact cosine;
- `0.2 * base MSE` and `0.2 * positive branch CE` through gradient reversal;
- frozen width-128 nonlinear probes trained for at most 240 steps on disjoint
  N3 interpolation probe splits;
- exact-pair test sets of 1,024 examples for interpolation, composition, and
  outside-range extrapolation.

## Primary endpoints

| Cut and shift | Ordinary `(cosine, branch, log-loss gain)` | Controlled `(cosine, branch, log-loss gain)` | Controlled seed passes |
| --- | --- | --- | ---: |
| Post-MLP composition | `(0.975, 0.554, 0.018)` | `(0.972, 0.684, 0.157)` | 0/5 |
| Full composition | `(0.971, 0.597, 0.041)` | `(0.971, 0.636, 0.087)` | 0/5 |
| Post-MLP extrapolation | `(0.458, 0.501, -0.027)` | `(0.544, 0.519, -0.102)` | 0/5 |
| Full extrapolation | `(0.452, 0.525, -0.040)` | `(0.556, 0.554, -0.044)` | 0/5 |

Success required cosine at least 0.90, branch accuracy at most 0.55, and
conditional log-loss gain at most 0.02 in all four cells for the same seed.
Every extrapolation cell failed cosine in every seed. Controlled full-depth
extrapolation cosine ranged from 0.476 to 0.649; the mean paired improvement was
0.104, with a five-seed 95% t interval of `[-0.030, 0.238]`.

Composition base retention was unchanged at full depth (paired mean difference
`-0.0002`, 95% interval `[-0.0038, 0.0033]`). At post-MLP, controlled
composition branch accuracy ranged from 54.5% to 89.3%. Its paired mean increase
was 13.0 points, but the five-seed interval `[-3.2, 29.3]` is wide; the
preregistered all-endpoint seed gate, not this interval, determines the verdict.

## Task non-inferiority

| Regime | Mean controlled-minus-ordinary exact-bin accuracy | Seeds within -3 points |
| --- | ---: | ---: |
| In distribution | `+1.00` points | 4/5 |
| Composition | `-1.72` points | 4/5 |
| Extrapolation | `+2.58` points | 5/5 |

Only three seeds passed task non-inferiority jointly across all three regimes;
four were required. Seed 7 lost 5.08 composition points and seed 53 lost 7.32
in-distribution points. The intervention therefore also failed the seedwise task
control, although its mean task changes were small and extrapolation accuracy
improved descriptively.

## Mechanism

| Condition and shift | Post-attention `(cosine, branch)` | Post-MLP `(cosine, branch)` | Full `(cosine, branch)` |
| --- | --- | --- | --- |
| Ordinary composition | `(0.975, 0.764)` | `(0.975, 0.554)` | `(0.971, 0.597)` |
| Controlled composition | `(0.971, 0.849)` | `(0.972, 0.684)` | `(0.971, 0.636)` |
| Ordinary extrapolation | `(0.460, 0.576)` | `(0.458, 0.501)` | `(0.452, 0.525)` |
| Controlled extrapolation | `(0.555, 0.610)` | `(0.544, 0.519)` | `(0.556, 0.554)` |

The MLP still suppresses attention-exposed branch information, but the
controlled circuit exposes and retains more compositional branch signal than
the ordinary circuit. Under extrapolation, the base improvement is already
present after attention; the MLP does not turn it into a high-fidelity base.

At the final training step, the temporary adversary averaged CE 0.692 and
online accuracy 50.3%, apparently near chance, while the fresh frozen probe
recovered substantial compositional branch information. This discrepancy is
the important adversarial-training failure: fooling the jointly trained
adversary did not establish invariance to a fresh nonlinear decoder.

## Local finite-perturbation diagnostic

The diagnostic used identical nuisance and direction draws at both ends of the
opposite-branch chord and recorded nonzero token changes. Mean post-MLP
`Q_local` changed as follows:

| Regime | Ordinary | Controlled |
| --- | ---: | ---: |
| In distribution | 7.43 | 7.21 |
| Composition | 5.64 | 4.79 |
| Extrapolation | 2.82 | 2.89 |

The controlled composition ratio decreased by 0.85, with a paired 95% interval
of `[-1.32, -0.38]`; full-depth composition decreased by 1.12
`[-1.54, -0.70]`. Extrapolation ratios were effectively unchanged. This
secondary diagnostic agrees with the probe failure rather than rescuing it.
It remains a quantized finite-perturbation ratio under diagonal whitening, not
an analytic Jacobian or causal proof by itself.

## Interpretation and next experiment

The horizontal auxiliary loss provides limited extrapolation benefit, but the
vertical adversary is not a reliable quotient constructor. More residual-space
losses would test optimizer games rather than the geometric hypothesis. The
next intervention should put the symmetry in the sensor map itself—for example,
an equivariant encoder with a declared invariant readout—and compare it only
against ordinary N3 over the same five seeds and shifts.

## Artifacts and reproduction

- Raw aggregate: `data/experiments/tinyllm_block1_quotient_control/20260806_d8_code_frozen/campaign_results.json`
- Per-seed results and weights: `data/experiments/tinyllm_block1_quotient_control/20260806_d8_code_frozen/runs/`
- Experiment: `experiments/structure_net/tinyllm_block1_quotient_control.py`
- Focused tests: `tests/structure_net/test_tinyllm_block1_quotient_control.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache pixi run python -m \
  experiments.structure_net.tinyllm_block1_quotient_control \
  --gpus 1 --slots-per-gpu 0 --max-parallel 2 \
  --output data/experiments/tinyllm_block1_quotient_control/20260806_d8_code_frozen
```

Logical ordinals are PyTorch ordinals, not NVIDIA-SMI physical indices.
