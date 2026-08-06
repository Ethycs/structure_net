# TinyLLM Nuisance-Support Scaling

**Status:** BROAD COVERAGE IMPROVES BASE RETENTION; NUISANCE-INVARIANT QUOTIENT NOT CONFIRMED

**Date:** 2026-08-06

**Hypothesis:** `tinyllm-nuisance-invariant-internal-cosine-quotient-v1`

**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-nuisance-support-scaling-preregistration.md`

## Verdict

Broader nuisance coverage produces a large but non-monotonic improvement in
cosine-base transfer. It does not produce the preregistered nuisance-invariant
internal quotient.

At N3, every objective retained cosine on the held-out compositional family
with correlation 0.967--0.974. None erased the phase branch: full-depth branch
accuracy remained 59.3--64.1%, above the 55% gate. On outside-range
extrapolation the pattern reversed. Branch probes were near chance, but cosine
correlation was only 0.452--0.511, far below the 0.90 base-retention gate.

The two shifted failures are therefore distinct:

- **composition:** the quotient base survives, but the fiber does not collapse;
- **extrapolation:** the fiber is not recoverable, but the quotient base is lost.

No arm reached strong success, coverage-dependent success, or
objective-dependent success. The preregistered conclusion is **failure despite
broad coverage**.

## Campaign integrity

The primary-35 design completed all 35 cells with zero failures: five seeds of
ordinary training at N0--N3, plus five N3 seeds for discrete multi-exit,
continuous gate, and quotient-contrastive training. Ten ordinary N0/N1 cells
were reused and 25 cells were newly scheduled. Nineteen previously completed
factorial cells remain supplementary, giving 54 retained detailed result and
checkpoint pairs in total.

Every cell used a d8 TinyLLM with 50,964,992 parameters, 4,096 training
examples, 600 AdamW updates, and batch size 64. The final two-worker CUDA path
completed without failure after changing the algebraically equivalent
multi-depth backward implementation to release each depth graph immediately.

## Ordinary nuisance scaling

| Training support | Early composition `(base, branch)` | Full composition `(base, branch)` | Early extrapolation `(base, branch)` | Full extrapolation `(base, branch)` |
| --- | ---: | ---: | ---: | ---: |
| N0 | (.000, 79.0%) | (.226, 60.1%) | (-.033, 62.9%) | (.079, 50.7%) |
| N1 | (.174, 80.0%) | (.251, 79.9%) | (-.006, 62.8%) | (.040, 58.3%) |
| N2 | (.002, 94.7%) | (.004, 87.8%) | (-.057, 78.0%) | (-.041, 63.8%) |
| N3 | **(.974, 69.1%)** | **(.971, 59.7%)** | **(.456, 56.0%)** | **(.452, 52.5%)** |

The paired N0--N3 cosine slope was positive in every seed at both primary cuts
and on both shifted regimes. N3 improved over N0 by 0.974 early and 0.745 full
on composition, and by 0.489 early and 0.373 full on extrapolation. This is not
a monotonic support-by-support law: N2 was worse than N1 and the gain arrived
as an N3 transition. It also fails the coverage-success definition because N3
composition retained the branch and N3 extrapolation did not retain cosine.

## N3 objective comparison

Full-depth means across five seeds:

| Objective | Composition base | Composition branch | Composition Fisher gamma | Extrapolation base | Extrapolation branch | Extrapolation Fisher gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary final | .971 | 59.7% | 1.012 | .452 | 52.5% | 1.067 |
| Discrete multi-exit | **.974** | 62.5% | .984 | **.511** | 52.5% | 1.030 |
| Continuous gate | .967 | 64.1% | 1.022 | .465 | 50.6% | 1.047 |
| Quotient contrastive | .973 | **59.3%** | 1.008 | .480 | **46.4%** | 1.031 |

The explicit quotient objective slightly reduced mean branch leakage, most
clearly on extrapolation, but did not cross the composition threshold. Joint
depth did not solve either failure. All N3 models learned the task on their
training support: mean exact-bin accuracy was 50.5--53.2% on a 16-bin output,
and interpolation cosine probes scored 0.985--0.988. The shifted failure is not
explained by an untrained model.

At full extrapolation, every ordinary seed retained only 0.324--0.570 cosine
correlation. Quotient-contrastive seeds ranged from 0.381 to 0.547. The failure
is therefore repeatable rather than a single bad initialization.

## Block-1 mechanism under shift

The earlier attention/MLP circuit survives qualitatively under N3 composition.
For ordinary training, post-attention branch accuracy was 76.4% and post-MLP
accuracy fell to 55.4%, while cosine correlation remained approximately 0.975.
The MLP again removes most of an attention-exposed nonlinear branch signal.

Under extrapolation, ordinary post-attention branch accuracy fell from 57.6%
to 50.1% after the MLP, but cosine correlation stayed near 0.46. This localizes
the extrapolation problem as base failure rather than failed fiber erasure.
The other objectives show the same broad separation: composition preserves the
base but leaks branch, while extrapolation suppresses branch without preserving
the task coordinate.

## Fiber geometry result

The branch-conditioned distance ratio did not repair the earlier MST proxy's
sensitivity problem. Fisher, Euclidean, and cosine median gamma values stayed
close to one across N3 cuts and regimes, including cells where nonlinear branch
accuracy exceeded 80%. For example, discrete post-attention composition had
82.4% branch accuracy while mean Fisher, Euclidean, and cosine gamma were
1.025, 1.019, and 1.023.

This directly falsifies the intended operational reading that gamma near one
is sufficient evidence of fiber identification. Average cross/within distance
is dominated by nuisance spread and can miss a low-dimensional but highly
decodable branch direction. Task-posterior Fisher distance is additionally
decoder-conditioned and can erase information that remains in the residual.
Gamma remains a descriptive control; it is removed from future headline
quotient claims unless paired with a sensitive local or direction-aware test.

## Preregistered gates

| Gate | Result |
| --- | --- |
| Strong success | Failed for all four objectives |
| Coverage-dependent success | Failed |
| Objective-dependent success | Failed |
| Failure despite broad coverage | **Passed** |
| Five independent seeds | Passed |
| Fixed samples, updates, and architecture | Passed |
| Composition split with familiar N2/N3 marginals | Passed |
| Checkpoint and detailed result retention | Passed |

## Interpretation and next intervention

The experiment does not support a nuisance-invariant semantic quotient. It
does show that data coverage can create a sharp transition in compositional
base retention, and that block-1 MLP computation continues to suppress a
branch signal. More nuisance sampling or stronger TDA summaries are not the
next justified move.

The next targeted intervention should encode or enforce a stable task
coordinate directly: an equivariant/invariant sensor encoder, or a local
block-1 objective that preserves cosine while adversarially removing the
conditional branch direction. That study should compare only ordinary N3 and
one invariant intervention over five seeds, retaining composition and
extrapolation as separate endpoints.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full/campaign_results.json` | primary-35 aggregate, scaling curves, gates, and scheduler record |
| `.../runs/<support>/<arm>/seed_<seed>/result.json` | frozen probes, block-1 cuts, fiber geometry, timing, and provenance |
| `.../runs/<support>/<arm>/seed_<seed>/model.pt` | retained final model weights |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python experiments/structure_net/tinyllm_nuisance_support_scaling.py \
  --design primary35 --seeds 7,17,29,41,53 \
  --steps 600 --train-samples 4096 \
  --slots-per-gpu 2 --max-parallel 2 \
  --reuse-existing-cells \
  --output data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full
```

Shared wall times are not performance benchmarks. Frozen nonlinear probes
bound tested decodability rather than conditional mutual information, and none
of the finite fiber measurements is a certified Reeb graph or cosheaf.
