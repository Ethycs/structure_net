# TinyLLM dynamic TTNO rank pilot

**Status:** MEASURED — BOND GROWTH OBSERVED PILOT
**Date:** 2026-08-29
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED EXPLORATORY PILOT`
**Hypothesis:** `tinyllm-dynamic-ttno-rank-pilot-v1`
**Preregistration:** [2026-08-29_tinyllm-dynamic-ttno-rank-pilot-
One judgment call I made: I did not move the scrapers out of event_harvester (that's Stage 4, gated on the "tests green with native deps absent" proof) — so event_harvester stays working. The submodule carries them as documented stubs, not preregistration.md](../07%20-%20Status%20Reports/2026-08-29_tinyllm-dynamic-ttno-rank-pilot-preregistration.md)

## Verdict
The declared bounded-bond TTNO criterion failed at 256 tokens. All 120 frozen
TinyLLM layer/head/input cells stayed inside the `ceil(log2(n)^2)` quantized
TTNO rank envelope through length 128, but only 42/120 (`35%`) did so at length
256, versus the preregistered `80%` requirement. Median 1%-error TTNO rank rose
from `31` at length 64 to `90` at length 256, and normalized rank increased from
`0.861` to `1.406` rather than remaining nonincreasing. The machine-readable
classification is therefore `bond_growth_observed_pilot`.

The narrower hierarchical-operator possibility remains open. At length 256 the
chronological token-tree HSS boundary rank had median `33`, far below the
quantized TTNO median of `90`, and natural attention was lower-rank than the
matched IID-Q/K control (`90` versus `135`). The pilot rejects the selected
strict quantized-TTNO envelope and trees for most cells; it does not reject an
H2 implementation, a different input-derived tree, or a learned compiler.

## Campaign integrity

| Item | Result |
| --- | ---: |
| requested evaluation seeds | 5 |
| completed | 5 |
| failed / excluded / retried | 0 / 0 / 0 |
| reused in producing launch | 0 |
| frozen checkpoints | 1 |
| selected cells per prefix | 8 layers x 3 heads x 5 prefixes = 120 |
| natural operator/length records | 480 |
| controls | causal uniform, smooth Fourier, IID Q/K |
| device | CPU |

The seeds `101, 211, 307, 401, 503` selected validation-stream prefixes. They
are input replicates, not independently trained models. Every raw record's
scientific fingerprint and implementation digest was checked against the
campaign record. Recomputing the aggregate from the five raw `result.json`
files reproduced `campaign_results.json` exactly.

The first full launch was interrupted before any seed record completed because
direct SVD made the rank sweep impractically slow. The preregistration records
the pre-outcome amendment: the 1% and 0.1% ranks were then computed from the
eigenvalues of the smaller Gram matrix, which are algebraically the squared
singular values. Direct SVD was retained for the `1e-12` sparse-rank diagnostic.
Focused tests confirmed identical declared-tolerance ranks before the producing
campaign was relaunched under a new implementation hash.

## Primary endpoints

Ranks below use the better of the two declared Q/K-only trees separately in
each frozen cell. The 0.1% column is a sensitivity diagnostic.

| Length | 1% TTNO median | Range | Envelope | Cell pass fraction | 0.1% TTNO median | 1% HSS median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 11–16 | 25 | 100% | 16 | 10 |
| 64 | 31 | 16–36 | 36 | 100% | 36 | 16 |
| 128 | 32 | 16–36 | 49 | 100% | 36 | 24 |
| 256 | 90 | 30–133 | 64 | 35% | 127 | 33 |

The discontinuity between 128 and 256 is not a rank-cap artifact: the 1%
median rose to `90`, while tightening tolerance to 0.1% raised it further to
`127`.

## Preregistered gates

| Gate | Requirement | Measured | Result |
| --- | --- | ---: | --- |
| Gaussian identity | max absolute error `<= 1e-10` | `3.44e-15` | pass |
| polylog envelope | at least 80% of cells pass at every length | `35%` | **fail** |
| normalized growth | median `rank/log2(n)^2` at 256 no larger than at 64 | `1.406 > 0.861` | **fail** |
| random separation | secondary: natural/IID median rank ratio `<= 0.75` | `90/135 = 0.667` | pass, secondary |

The two primary rank gates both failed. The secondary random-control result
does not rescue the classification.

## Tree dependence

The input-derived PCA ordering did not find a better tree. At length 256,
chronological order had lower TTNO rank in all 120 cells:

| Ordering | TTNO median | Range | HSS boundary median |
| --- | ---: | ---: | ---: |
| chronological | 90 | 30–133 | 33 |
| Q/K PCA | 144.5 | 44–227 | 47 |

Thus the optimistic `best_declared_tree` endpoint always selected the
chronological tree. This is evidence against this particular Q/K-only tree
generator, not against all input-derived cluster trees.

## Layer and head localization

The 256-token bond growth is structured by depth rather than confined to one
input or head. Every evaluation seed had a median between `85.0` and `92.5` and
an envelope pass fraction between `33.3%` and `37.5%`.

| Layer | Median TTNO rank | Range | Envelope pass fraction |
| ---: | ---: | ---: | ---: |
| 0 | 115 | 90–122 | 0% |
| 1 | 87 | 80–110 | 0% |
| 2 | 84 | 31–125 | 33.3% |
| 3 | 40 | 30–53 | 100% |
| 4 | 41 | 33–75 | 80% |
| 5 | 53 | 47–98 | 66.7% |
| 6 | 110 | 99–125 | 0% |
| 7 | 126 | 116–133 | 0% |

Middle layers 3–5 are substantially more compatible with the declared TTNO
envelope than early and late layers. Head medians were `86.5`, `85.0`, and
`109.0` for heads 0, 3, and 7 respectively, so no selected head family supplies
a global low-rank explanation.

## Controls and sparse exceptions

The controls show that the estimator distinguishes known structured operators
from generic causal attention:

| Condition | Rank at 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: |
| causal uniform | 4 | 6 | 7 | 7 |
| smooth Fourier | 4 | 6 | 8 | 8 |
| natural TinyLLM | 16 | 31 | 32 | 90 |
| IID Q/K | 16 | 36 | 36 | 135 |

Natural attention is neither a bounded-rank positive control nor fully generic:
its median 256-token rank is one third below IID Q/K, but roughly eleven times
the smooth control.

Separating the top two allowed attention entries per row did not produce a
lower-rank remainder. Under the chronological tree, those entries captured a
median `40.72%` of total attention mass, but remainder rank rose from median
`90` to `111`; it improved in `0/120` cells. The sparse component itself had
median exact cut rank `47`. Across both orderings, only `2.5%` of remainders
improved, matching the aggregate secondary diagnostic. A useful sparse-TTNO
implementation would therefore need a more structural exception rule than
row-wise top-two magnitude.

## Interpretation

This pilot cleanly separates three statements:

1. **Exact algebra:** causal softmax attention matched the normalized Gaussian
   reconstruction to `3.44e-15`; the identity is not the obstruction.
2. **Token-tree hierarchy:** natural attention retained moderate HSS boundary
   ranks, particularly in chronological order, so hierarchical operator
   compression remains plausible.
3. **Strict quantized TTNO:** the declared paired-bit TTNO ranks grew beyond the
   finite polylog envelope for most 256-token cells. Local or token-tree
   low-rank structure did not automatically compile into small global TTNO
   bonds.

The measured result therefore supports the motivating note's warning that
individual hierarchical low-rank interactions are insufficient: the global
tree-cut condition must be checked. Under the two checked trees, that condition
fails for most cells at the longest available context.

## Boundaries and next experiment

- The study uses one d8 pretraining checkpoint and cannot estimate model-seed
  variation.
- Four context lengths ending at 256 cannot establish an asymptotic law.
- The rank sweep materializes dense attention and does not test core-generation
  or application complexity.
- A quantized TTNO over `log2(n)` paired index bits is not a many-body TTNO with
  one physical tensor-product site per token.
- HSS ranks are a strong token-tree boundary diagnostic, not a constructed H2
  approximation with row-relative error control.
- Numerical cut ranks do not by themselves produce one simultaneous
  epsilon-accurate TTNO.

The next discriminating experiment should preserve the chronological token
tree and construct an actual nested H2 approximation with a held-out output
error certificate. Only after that succeeds should it test a compiler into a
strict operator network, comparing: (a) the present paired-bit TTNO, (b) an
HSS-derived operator network, and (c) a learned tree optimized on training
prefixes but frozen before evaluation. It should repeat across model seeds and
extend context length beyond 256.

## Artifacts and reproduction

- Aggregate: `data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot/campaign_results.json`
- Raw cells: `data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot/runs/seed_*/result.json`
- Runner: `experiments/structure_net/tinyllm_dynamic_ttno_rank.py`
- Tests: `tests/structure_net/test_tinyllm_dynamic_ttno_rank.py`

```bash
pixi run pytest -q tests/structure_net/test_tinyllm_dynamic_ttno_rank.py

pixi run python -m experiments.structure_net.tinyllm_dynamic_ttno_rank \
  --device cpu \
  --output data/experiments/tinyllm_dynamic_ttno_rank/20260829_d8_babylm_pilot
```

The producing campaign used implementation digest
`ffdf9bb77449a4dcad6c67f111b70a3543eae42495ab067044835d13bf65c8fb`.
