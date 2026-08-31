# TinyLLM constructive causal H2 attention result

**Status:** COMPLETED
**Hypothesis:** `tinyllm-causal-h2-attention-v1`
**Classification:** `h2_representation_pass_no_finite_size_compression`
**Campaign artifact:** `data/experiments/tinyllm_causal_h2_attention/20260830_registered/campaign_results.json`
**Campaign SHA-256:** `4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2`

## Outcome

A7 constructed the preregistered strong-admissibility H2 approximation for all
five frozen prefixes, eight layers, three heads, and four lengths on GPU 0. All
480 cell-length records passed the numerical construction contract: partition
fingerprints, stabilized-softmax replay, basis orthogonality, parent-child
nestedness, explicit-versus-contracted matvec, causal leakage, and finiteness.

### Representation gates

| Length | Passing cells | Pass fraction | Median maximum kernel-row error | p90 |
| ---: | ---: | ---: | ---: | ---: |
| 32 (integrity) | 120/120 | 100% | 0 | 0 |
| 64 | 120/120 | 100% | `4.45e-4` | `1.60e-3` |
| 128 | 120/120 | 100% | `1.13e-3` | `2.79e-3` |
| 256 | 118/120 | 98.33% | `1.64e-3` | `5.95e-3` |

Every primary length exceeded the required 80% pass fraction. Layers 0–6
passed all length-256 cells; layer 7 passed 86.67%, above the required 50%.
The direct H2(A) oracle passed in every cell.

The two primary cell failures were both layer 7, head 3:

- seed 307: kernel row-relative error `0.01384`;
- seed 401: kernel row-relative error `0.01024`.

Only the 1% kernel-row certificate failed in those cells. Their normalized
attention, random-probe, learned-value output, and token-tail errors still
passed. This localizes the issue to the conservative unnormalized-kernel gate,
not a functional output failure.

### Finite-size compression gate

| Length-256 cost ratio | Median | p90 | Required |
| --- | ---: | ---: | ---: |
| stored H2 scalars / dense causal scalars | 1.823 | 2.063 | `<= 0.75`, `<= 1.0` |
| H2 / dense theoretical multiply-adds | 1.764 | 2.005 | `<= 0.75`, `<= 1.0` |

The representation therefore passes, but the fixed overhead from leaf bases,
transfers, couplings, and exact near blocks is larger than dense causal storage
and arithmetic at 256 tokens. It cannot receive
`h2_constructive_compression_pass` at this finite size.

## Interpretation

Together, A6 and A7 show that the low chronological boundary ranks can be
assembled into a single accurate shared and nested operator. This is a positive
answer to the representation question and a negative answer to finite-size
efficiency under the locked `b=16`, one-diameter admissibility, and polylog rank
cap.

The defensible claim is therefore:

> Frozen TinyLLM causal attention admits the tested normalization-faithful H2
> representation at 64–256 tokens, but that representation does not compress
> storage or theoretical work at 256 tokens.

A7 materialized dense operators to construct the bases. It does not provide
A9's implicit core compiler, establish an asymptotic crossover, or demonstrate
wall-clock advantage. The frozen checkpoint cannot test the conditional
512-token extension.

## Non-rescuing sensitivities

These whole-campaign arms ran only after the primary checksum was frozen. None
can alter the primary verdict.

| Arm | Length-256 pass | Median storage ratio | Median operation ratio | Result |
| --- | ---: | ---: | ---: | --- |
| leaf size 8 | 100% | 1.906 | 1.878 | representation pass, no compression |
| leaf size 32 | 96.67% | 1.612 | 1.491 | representation pass, no compression |
| separation ratio 0.5 | 98.33% | 1.823 | 1.764 | same as primary |
| separation ratio 2.0 | 100% | 1.556 | 1.498 | representation pass, no compression |
| rank envelope 0.5x | 61.67% | 1.492 | 1.434 | representation failed |
| rank envelope 2x | 98.33% | 1.823 | 1.764 | same as primary |

The result is robust to larger rank allowance, leaf size, and stricter
separation. A half-size rank envelope is insufficient, while every tested arm
still costs more than dense attention at 256 tokens. Direct H2(A) was already
included in every primary cell. The learned contiguous-tree arm remains a
separate later campaign, as preregistered.
