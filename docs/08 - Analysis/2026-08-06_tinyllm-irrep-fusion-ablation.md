# TinyLLM deck-irrep fusion ablation

**Status:** PARTIALLY SUPPORTED — CHARGED CARRIERS ARE CAUSAL; A UNIVERSAL FINITE-`C3` PHASE MECHANISM FAILS  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-deck-irrep-fusion-ablation-v1`  
**Preregistration:** [`2026-08-06_tinyllm-irrep-fusion-ablation-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-irrep-fusion-ablation-preregistration.md)

## Verdict

Applying the exact deck symmetry exposes a causal coordinate system, but not
one universal fusion law. At every frozen synthesis front for both `C2` and
`C3`, removing all nontrivial characters made the continuation fail and
restoring the exact carrier made it pass. This charged-mode necessity gate
passed in 5/5 seeds under both composition and extrapolation for both degrees.
The norm-matched cross-orbit carrier did not meet the joint reproduction
criterion in any seed.

The stronger degree-three prediction failed. Continuous rotations of the real
`C3` carrier, which preserve the barycenter and the quadratic radial invariant
`c1*c2=|c1|^2` while changing discrete-phase invariants, had a large causal
effect in seeds 17, 29, and 53. Seed 41 was mixed under both shifts, and seed 7
changed from mixed on composition to phase-sensitive on extrapolation. The
finite-`C3` phase gate therefore reached only 3/5 seeds instead of four, while
the phase phenotype itself was shift-stable in 4/5.

The supported architectural conclusion is narrower:

> Preserve typed deck-character carriers and permit every symmetry-allowed
> neutral fusion. Do not replace `C3` by an `O(2)` radial norm, but do not assume
> that cubic/discrete-phase fusion dominates every checkpoint either.

## Campaign integrity

The campaign reused ten frozen d6 checkpoints and the first synthesis
transition frozen by the predecessor character-coupling campaign. It performed
no training, front selection, probe fitting, or parameter update. Each cell
regenerated the predecessor's same 64 exact orbits so that the intervention
could decompose the already-established event rather than test front
replication. Consequently, charged-mode necessity here is within-cohort causal
evidence, not an independent replication claim.

Both `C2` and `C3` completed a separate eight-orbit CUDA lifecycle before the
primary run. That shakedown exposed and repaired a float32 identity-distance
artifact before primary outcomes were inspected. Five algebraic and numerical
contract tests passed after the repair.

| Item | Value |
| --- | --- |
| requested / completed / failed | 10 / 10 / 0 |
| independently trained checkpoints | five each for `k=2,3` |
| transitions evaluated | one frozen synthesis transition per shift and checkpoint |
| exact orbits | 64 per shift and cell |
| training or fitted observers | none |
| amplitude grid | `0, 0.25, 0.50, 0.75, 1.00` |
| `C3` phase grid | 12 angles on `[0,2 pi)` |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| total analysis time | 34.6 seconds |
| implementation SHA-256 | `4b2dd01bd61bfb97dac4c407a34f2f562ded26f696c2da80daf64be5b3f1d6fa` |
| campaign SHA-256 | `bc1b60ea2fbda513a042c533d33921ced13a64005958ebd416ca7690bd890065` |
| DVC data root | `bf9179b07bea441fd66ba63ec6543b44.dir` |
| lakeFS commit | `f6a911dbf7d49ff6f31e6d96f0c0c96b3602ee88117b21bbc01f18cb4b6e3255` |

## Preregistered gates

Each gate required at least four of five seeds, jointly across composition and
extrapolation.

| Gate | `k=2` | `k=3` | Required |
| --- | ---: | ---: | ---: |
| exact group contract | **5/5** | **5/5** | 4/5 |
| charged-mode necessity | **5/5** | **5/5** | 4/5 |
| substituted carrier does not reproduce | **5/5** | **5/5** | 4/5 |
| finite-`C3` phase mechanism | n/a | 3/5 | 4/5 |
| shift-stable phase phenotype | n/a | **4/5** | 4/5 |

The full hypothesis is not confirmed because the primary finite-`C3` phase
gate missed the replication threshold.

## Exact group and carrier controls

True deck rotations only permute the input sheets. Across all evaluated cells,
the largest relative next-barycenter error was exactly zero for `C2` and
`3.56e-8` for `C3`; the largest downstream Fisher--Rao discrepancy was
`1.22e-13`. These are well inside the frozen group-contract tolerances.

The cross-orbit substitution retained each barycenter, the complete real deck
carrier structure, and its norm while breaking the carrier/base pairing. It
never met the joint reproduction criterion. No `C3` substituted patch passed
the task gate. The `C2` seed-17 control did pass that coarse task gate under
both shifts, but preserved only `0.394` and `0.590` of the exact Fisher effect,
below the frozen `0.70` reproduction threshold. Thus the 5/5 specificity result
does not mean every substituted carrier is task-invalid; it means none
reproduces the declared exact mechanism.

Median substituted-carrier Fisher preservation was `0.119 / 0.082` for `C2`
and `-0.615 / -0.012` for `C3` on composition / extrapolation. Negative values
mean the substitution moved the output farther from the exact posterior than
deleting the carrier entirely.

## Degree-three phase intervention

Normalized sensitivity can exceed one: it means rotating the carrier changed
the posterior more than removing that carrier. The table reports composition /
extrapolation at the frozen synthesis transitions.

| Seed | Target cut, composition / extrapolation | Phase sensitivity | Phenotype | Shift-stable |
| ---: | --- | --- | --- | --- |
| 7 | block-3 attention / block-1 attention | `0.118 / 0.396` | mixed / phase-sensitive | no |
| 17 | block-1 attention / block-0 attention | `1.947 / 1.109` | phase-sensitive / phase-sensitive | yes |
| 29 | block-1 attention / block-1 attention | `1.884 / 1.342` | phase-sensitive / phase-sensitive | yes |
| 41 | block-0 MLP / block-0 MLP | `0.242 / 0.193` | mixed / mixed | yes |
| 53 | block-1 attention / block-1 attention | `0.405 / 0.412` | phase-sensitive / phase-sensitive | yes |

This rules out a purely radial `O(2)` surrogate for three of five checkpoints:
their task-effective invariant depends on information changed by continuous
phase rotation but preserved by the discrete `C3` action. Because two seeds do
not replicate that phenotype, the data instead support a mixture of allowed
invariants. Candidate generators include the radial quadratic term
`|c1|^2` and finite-group phase terms such as `Re(c1^3)` and `Im(c1^3)`.
The intervention identifies their functional distinction but does not uniquely
fit those polynomial coordinates.

## Amplitude and quadratic-chord diagnostic

The first task-valid exact-carrier amplitudes were:

| Degree | Seed | Composition | Extrapolation |
| --- | ---: | ---: | ---: |
| `C2` | 7 | 0.75 | 0.75 |
| `C2` | 17 | 0.75 | 0.75 |
| `C2` | 29 | 0.75 | 0.50 |
| `C2` | 41 | 1.00 | 0.75 |
| `C2` | 53 | 0.75 | 0.75 |
| `C3` | 7 | 0.50 | 0.75 |
| `C3` | 17 | 0.75 | 0.75 |
| `C3` | 29 | 1.00 | 1.00 |
| `C3` | 41 | 0.75 | 1.00 |
| `C3` | 53 | 1.00 | 0.75 |

At amplitude `0.75`, the non-derivative quadratic chord
`a + alpha^2(q(1)-a)` explained median downstream effects of `0.988 / 0.980`
for `C2` and `0.867 / 0.799` for `C3`. At amplitude `0.50`, the corresponding
medians were `0.768 / 0.665` and `0.628 / 0.379`.

This does not rescue the failed local-Hessian result. The chord is given the
observed full-radius defect and tests only how that defect scales back along
one ray; the earlier HVP attempted to predict the full defect from derivatives
at the barycenter. The present result says that a full-effect direction often
becomes a useful approximation near the causal threshold. It does not say a
universal quadratic operator generated that direction.

## Relation to fresh-orbit radius evidence

The separate [causal orbit-radius titration](2026-08-06_tinyllm-orbit-radius-titration.md)
is the stronger out-of-sample test of degree-two radial stability. It used new
orbits and found a single `C2` threshold in all five seeds, with shift-stable
radius in four. The independent [degree-two fusion-radius study](2026-08-06_tinyllm-c2-fusion-radius.md)
also found direction specificity but exposed cohort-sensitive hard-gate
exceptions. Together with this same-cohort decomposition, the conservative
conclusion is that `C2` carrier amplitude is a useful causal coordinate, while
its exact threshold remains evaluation- and decoder-sensitive.

## Architectural consequence

An equivariant replacement should carry the trivial and charged irreps through
the network and expose only symmetry-neutral fusion channels to the invariant
readout:

```text
C2: c1 x c1 -> c0, with an amplitude-aware radial gate
C3: c1 x c2 -> c0, plus permitted c1^3 and c2^3 phase channels
```

Projecting to the trivial irrep before these interactions would delete a
causally necessary carrier in every tested checkpoint. Conversely, enforcing
continuous `O(2)` invariance would suppress the discrete-phase channels used
strongly by three `C3` seeds. The next architecture should therefore implement
exact `Ck` equivariance, not generic rotation invariance, while allowing the
learned readout to weight radial and finite-phase invariants differently.

## Boundaries

Continuous `C3` phase rotations are off-orbit interventions inside the real
isotypic carrier. They preserve declared group statistics but need not lie on
the natural activation support. Sensitivity therefore establishes causal use
of carrier phase by the frozen continuation, not a global representation
theorem. The decoder-conditioned causal gate and the same-cohort front remain
important limitations.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered/campaign_results.json`
- Per-cell records:
  `data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered/runs/k{2,3}/seed_<seed>/result.json`
- Disposable CUDA lifecycle:
  `data/experiments/tinyllm_irrep_fusion_ablation/shakedown_20260806/`
- Runner: `experiments/structure_net/tinyllm_irrep_fusion_ablation.py`
- Tests: `tests/structure_net/test_tinyllm_irrep_fusion_ablation.py`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_irrep_fusion_ablation \
  --output data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered \
  --device cuda:0
```
