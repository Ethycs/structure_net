# Independent Qwen block failures: development diagnosis

Study date: 2026-11-14. Exploratory, read-only analysis of existing development
artifacts; no training, GPU work, or confirmation-set access.

Blocks12 and15 missed the isolated47/64 gate, scoring44/64 and46/64. Their
aggregate projection floors and remaining fitting errors do not distinguish
them from nearby successful blocks. Improving feature reconstruction also did
not consistently improve task accuracy. Assembly must be measured separately.

## Evidence and scope

The completed34-block campaign is
`data/experiments/jspace_parallel_single_layer/20261114_v1/result.json`.
It reports34 verified selections,32 isolated passes, and misses at12/15.
For each layer below, the sources are
`blocks/block_NN/train/result.json` and its
`epoch_XX_development.json` files under that campaign root. Fresh verification
receipts are in each block's `verification/` directory.

The native reference is
`data/experiments/jspace_native_dev_cache/20261114_v1/cache/result.json`:
50/64 on the same ordered development identities. There are16 cases each from
ARC, BoolQ, HellaSwag, and WinoGrande. No fresh512-case confirmation was read.

## Feature reconstruction versus accuracy

“Floor” is the fixed outside-basis component of teacher-update-relative MSE;
“retained error” is total relative MSE minus that floor. Both use the existing
equal-case, equal-valid-token weighting and native teacher update-energy
denominator. These are feature-space quantities, not task-accuracy bounds.

| Block | Initial correct | Selected epoch | Selected correct | Final epoch12 correct | Outside floor | Selected retained error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 44 | 3 | 44 | 42 | .258545 | .107029 |
| 14 | 45 | 2 | 50 | 47 | .245381 | .121469 |
| 15 | 40 | 12 | 46 | 46 | .238514 | .103132 |
| 16 | 45 | 11 | 50 | 49 | .231372 | .118637 |

The failed blocks have **less retained feature error at selection** than these
two passing controls. Their floors are not exceptional either: passing
block11 has .264923, and passing block13 has .257291. Blocks6–10 also pass
with floors above block12's.

Retained error fell65.8% for block12 and71.4% for block15, compared with62.2%
and67.1% for blocks14/16. Thus substantial fitting occurred. At the selected
checkpoints, fixed floors account for70.7%/69.8% of total loss in12/15 and
66.9%/66.1% in14/16; this does not establish a task-performance ceiling.

Block12's total relative MSE keeps improving from .365574 at its selected
epoch3 to .346525 at epoch12 while accuracy falls44→42. Block14 likewise
improves .366850→.333789 while accuracy falls50→47. Simply extending the
same feature objective is not supported as a reliable accuracy fix.

## Which answers change

| Block | Gains over native | Losses against native | ARC | BoolQ | HellaSwag | WinoGrande |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | — | — | 15 | 13 | 14 | 8 |
| 12 | 2 | 8 | 12 | 12 | 10 | 10 |
| 14 | 4 | 4 | 14 | 14 | 12 | 10 |
| 15 | 1 | 5 | 14 | 13 | 12 | 7 |
| 16 | 4 | 4 | 13 | 14 | 12 | 11 |

Passing at50/64 does not mean reproducing the native answers: blocks14/16
each exchange four native successes for four gains. This matters when judging
whether independently fitted blocks will compose successfully.

The failures include confident reversals, not only nearly tied choices. For
the ARC question asking why an atom acquires charge−1, the labeled answer is
gaining one electron. Its true-answer margin is+13.28 for native,−14.22 for
block12, and−15.98 for block15. Identity:
`704d6cb0548d6f8f71c170ad9df2c0d1b320907ddca56eec5742ddc6d7ecbb1f`.
Here margin means the correct answer's logit minus the largest alternative.
Neither failed model exhibits a constant-label collapse.

## BF16 training context versus FP32 quality context

A CPU-only comparison used original BF16 boundary banks from
`data/experiments/jspace_full_compact/20261113_capture_v1/capture/`
and corresponding FP32 banks from
`data/experiments/jspace_native_dev_cache/20261114_v1/cache/`.
Only the64 development rows were compared, using original offsets in
`data/experiments/jspace_quadratic_recipe/20261114_v1/items.json`.

| Block input | Relative input RMS difference | Relative native-update MSE difference |
| --- | ---: | ---: |
| 12 | 2.896% | .003296 |
| 14 | 2.570% | .004170 |
| 15 | 2.444% | .003656 |
| 16 | 2.411% | .001914 |

The input statistic is100 times the square root of the equal-case/token mean
of squared input difference divided by FP32 input energy. The update statistic
compares the two native updates using the same original BF16 update-energy
denominator and training-derived floor as the corresponding worker.

This distribution difference exists, but does not spike uniquely at the failed
blocks. The cached original BF16 teacher's final choice logits score48/64,
versus50/64 for FP32 native. Its two lost cases overlap only one of block12's
eight native-relative losses and none of block15's five. These observations do
not prove that precision/context mismatch is harmless or that it caused the
remaining failures; small state changes can affect sensitive directions.

## Smallest next diagnostic, after assembly

First complete the declared native/accepted-two/all-compact/hybrid assembly
comparison. If repair is needed, use the existing native FP32 boundaries to
score the projected-update surrogate
`h_l + U Uᵀ(h_{l+1} − h_l)` for blocks12/15 through their original suffixes.
Also measure the selected blocks' local feature errors against those same FP32
targets. This changes neither weights nor architecture and distinguishes a
poor reachable subspace from a poor learned realization more directly than
aggregate BF16 feature loss.

If those surrogates preserve task quality but the learned blocks do not, add
matched native-versus-compact suffix controls under BF16 and FP32 input
boundaries before attributing the gap to precision or insufficient fitting.
The projection surrogate is **not an upper bound on task accuracy**. No such
diagnostic has been run here, and these64 repeatedly used development cases
cannot establish generalization or statistical equivalence.
