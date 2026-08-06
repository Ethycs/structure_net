# TinyLLM Frozen Internal-Quotient Probes

**Status:** STRONG SINGLE-SEED ASYMMETRY; HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-05  
**Applies to:** `experiments/structure_net/tinyllm_internal_quotient_probe.py`  
**Hypothesis:** `tinyllm-internal-cosine-quotient-erases-branch-v1`  
**Depends on:** `2026-08-05_tinyllm-task-quotient-contrast.md`

## Measured verdict

Frozen layerwise probes support genuine internal quotient formation much more directly than posterior H1 alone. On exact cosine-matched phase pairs, the final residual of each phase-trained model retained the branch bit almost perfectly, while both cosine-trained models remained near the balanced 50% baseline. The asymmetry survived a disjoint third-harmonic, temporal-drift, Laplace-noise nuisance family.

Cross-decoding behaved as predicted: cosine was easy to recover from both representations, phase was easy from the phase representation, and phase remained ambiguous from the cosine representation. The separation appeared at the first transformer block and persisted through the final block.

Independent persistent cohomology found strong target-aligned circles in both phase-trained final residuals and only weak H1 in the cosine-trained residuals. The full preregistered saved-checkpoint criterion set nevertheless failed: d6 cosine's weak circular coordinate had alignment 0.6033, 0.0033 above the frozen 0.600 ceiling. More importantly, only seed-7 checkpoints were retained, so this experiment cannot confirm a repeatable multi-seed claim.

## Protocol

| Field | Value |
| --- | --- |
| Source models | saved d6/d8 phase-circle and cosine-interval seed-7 checkpoints |
| Source-model updates | none; every source parameter had `requires_grad=False` |
| Layers | constant final-query embedding baseline plus every transformer-block residual |
| Probe | separate two-hidden-layer nonlinear heads for branch, cosine, and phase vector |
| Probe selection | 4,096 train; 1,024 held-out validation with early stopping |
| Evaluation | 2,048 in-distribution and 2,048 disjoint-nuisance examples |
| Conditional branch | `sign(sin(future phase))`, balanced within exact equal-cosine pairs |
| Cross-decoders | residual to cosine; residual to `(cos phase, sin phase)` |
| Disjoint nuisance | third harmonic, temporal channel drift, Laplace noise, unseen parameterization |
| Independent topology | longest persistent H1 cocycle, finite-field lift, least-squares circular coordinate |
| Coordinate labels | phase labels used only after coordinate discovery, for alignment evaluation |
| Hardware | NVIDIA GeForce RTX 2060 SUPER; PyTorch 2.5.1+cu121 |

The branch probe also receives true cosine as a covariate. Because every cosine value occurs with both branches, cosine alone cannot beat 50%; the probe measures information remaining after conditioning on the quotient coordinate.

## Final-layer results

| Model | Quotient | Branch ID | Branch disjoint | Cosine Pearson ID | Phase alignment ID | Phase alignment disjoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d6 | phase | **99.90%** | **96.29%** | 0.988 | **0.995** | **0.873** |
| d6 | cosine | **48.63%** | **52.59%** | 0.986 | 0.471 | 0.364 |
| d8 | phase | **99.90%** | **95.70%** | 0.984 | **0.992** | **0.873** |
| d8 | cosine | **53.22%** | **50.34%** | 0.986 | 0.474 | 0.391 |

The 95% Wilson intervals for phase-model branch recovery are 99.64--99.97% in distribution and 94.74--97.02% on the disjoint family. The cosine-model intervals are 46.47--50.80% and 51.06--55.38% in distribution, and 50.42--54.74% and 48.18--52.50% on the disjoint family. Three of four cosine cells include 50%; d8 in distribution is slightly above it but far below the phase model and below the frozen 60% ceiling.

Disjoint-family cosine correlation fell to 0.727--0.820 for both source tasks. The nuisance family is therefore materially difficult, not a cosmetic resampling. Despite that degradation, the branch asymmetry remained large.

## Layerwise branch formation

| Model | Quotient | First block ID | Final block ID | First block disjoint | Final block disjoint |
| --- | --- | ---: | ---: | ---: | ---: |
| d6 | phase | 98.24% | 99.90% | 95.41% | 96.29% |
| d6 | cosine | 54.59% | 48.63% | 52.00% | 52.59% |
| d8 | phase | 99.02% | 99.90% | 94.78% | 95.70% |
| d8 | cosine | 51.17% | 53.22% | 50.39% | 50.34% |

The final-query embedding is constant across examples and correctly gives 50%; it is a negative baseline, not a representation of the whole input sequence. At the first contextual transformer block, the task-dependent asymmetry is already nearly complete. Phase-trained blocks expose the branch immediately, while cosine-trained blocks expose cosine but not the branch.

This supports a rapid task-conditioned quotient in the query residual stream. Probe failure cannot prove information-theoretic erasure, but the paired phase model demonstrates that the same probe family, inputs, and layer positions can recover the branch when training preserves it.

## Cross-decoding

The final phase-to-cosine direction is easy in both phase models: Pearson 0.988 for d6 and 0.984 for d8. Cosine-trained final residuals also decode cosine at 0.986 in both classes.

The reverse direction is asymmetric. Phase alignment from phase-trained residuals is 0.995/0.992, while cosine-trained residuals reach only 0.471/0.474. Direct branch accuracy gives the sharper conditional result above. Together these show that the cosine representation preserves the quotient coordinate while failing to expose which preimage branch generated it.

## Independent persistent cohomology

| Model | Quotient | Base H1 | Base coordinate alignment | Disjoint H1 | Disjoint coordinate alignment |
| --- | --- | ---: | ---: | ---: | ---: |
| d6 | phase | **0.688** | **0.977** | **0.459** | **0.804** |
| d6 | cosine | 0.106 | 0.603 | 0.201 | 0.303 |
| d8 | phase | **0.850** | **0.954** | **0.639** | 0.645 |
| d8 | cosine | 0.092 | 0.600 | 0.113 | 0.600 |

The coordinate is derived from Euclidean residual distances and the longest persistent cocycle; true phase enters only afterward to measure rotation/orientation-invariant alignment. This separates cycle discovery from supervised-map verification.

The base-family result is clear in persistence strength: phase H1 is 0.688--0.850 versus 0.092--0.106 for cosine. Alignment of a weak/noisy longest cocycle is not meaningful by itself, which is why d6's 0.603 alignment should be read alongside H1 0.106. The frozen executable criterion used alignment alone and therefore failed by 0.003; it has not been retroactively changed to an H1-or-alignment rule.

The disjoint result is less uniform. d6 retains an aligned circle, while d8 retains strong H1 but its recovered coordinate aligns only 0.645. The supervised phase probe still aligns 0.873, suggesting that independent coordinate recovery is more nuisance-sensitive than information availability.

## Preregistered saved-checkpoint criteria

| Criterion | d6 | d8 |
| --- | --- | --- |
| Cosine decodes from both final representations | pass | pass |
| Phase-model branch ID at least 80% | pass | pass |
| Phase-model branch disjoint at least 70% | pass | pass |
| Cosine-model branch ID at most 60% | pass | pass |
| Cosine-model branch disjoint at most 60% | pass | pass |
| Phase-to-phase alignment at least 0.9 | pass | pass |
| Cosine-to-phase alignment at most 0.6 | pass | pass |
| Independent phase coordinate alignment at least 0.8 | pass | pass |
| Cosine coordinate alignment at most 0.6 | **fail: 0.6033** | pass: 0.6000 |

The aggregate is `predicted_asymmetry_observed: false` and `confirmed: false`. The direct branch/cross-decoding subclaims are supported in both model classes, but the all-criteria conjunction is not.

## Follow-up atlas

The post-attention/post-MLP atlas has now been run; see `2026-08-05_tinyllm-layer-task-geometry-atlas.md`. It localizes phase carriers to attention outputs, conditional branch collapse to block 1 MLP, and mature cosine interval geometry to later attention stages in both retained model classes.

## Scope and next test

**Supported in these retained checkpoints:**

- internal conditional branch asymmetry between matched phase and cosine tasks;
- the asymmetry appears by the first block and survives a disjoint nuisance family;
- cross-decoding follows the expected function/preimage direction;
- independent cohomology recovers a strong phase circle without using the decoder.

**Not established:**

- repeatability across independently trained seeds;
- information-theoretic or causal erasure—the probe family may miss a code;
- robustness to all nuisance families;
- transfer to real sensors or pretrained language models.

The decisive replication is to retain and probe at least five independently trained checkpoint pairs. Add matched-capacity random-label probes, linear probes, and an intervention that attempts to flip the branch while holding decoded cosine fixed. This would distinguish nonlinear decodability, selectivity, and causal availability.

## Artifacts and reproduction

| Path | Contents |
| --- | --- |
| `data/experiments/tinyllm_internal_quotient_probe/20260805_d6_d8_seed7/results.json` | four checkpoint analyses, every-layer probe metrics, both nuisance families, cohomology coordinates |
| `data/meta_hypotheses/tinyllm-internal-cosine-quotient-erases-branch-v1.json` | conservative single-seed meta-hypothesis and four linked records |
| `data/chroma_db` | searchable hypothesis and linked experiment results |

The completed result and partial envelope are byte-identical with SHA-256 `5272db402ef509a670f9dea129ec4752c542783020e709a20eb1bf32407f1485`.

```bash
env MPLCONFIGDIR=/tmp/structure-net-matplotlib \
  pixi run python experiments/structure_net/tinyllm_internal_quotient_probe.py \
    --device cuda:auto \
    --probe-steps 300 \
    --train-samples 4096 \
    --validation-samples 1024 \
    --test-samples 2048 \
    --output data/experiments/tinyllm_internal_quotient_probe/20260805_d6_d8_seed7

pixi run python \
  experiments/neural_architecture_lab/store_internal_quotient_probe_meta_hypothesis.py
```

The focused analyzer, runner, and ledger gate completed with **18 passed, 0 failed** after the class-prior ledger assertions were added. After the sublayer atlas extension, the final full repository gate completed with **350 passed, 1 skipped, 0 failed** and 23 warnings in 386.19 seconds.
