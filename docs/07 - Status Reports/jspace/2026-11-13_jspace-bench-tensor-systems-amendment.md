# CPU counting fix for shared basis overlap

Date: 2026-11-13. Study: JSPACE-BENCH-TENSOR-01, systems revision 2.

The v1 CUDA shakedown and full calibration/development capture completed. The fit
stage failed at the first layer's reporting step because Torch 2.4.1 does not
implement integer matrix multiplication on CUDA. No fit manifest, candidate
selection, or held-out model evaluation was produced. The error and its lease are
preserved in `20261113_v1/fit_driver.log` and `20261113_v1/fit/lease.json`.

Move only the tiny Boolean eligibility arrays used by the redundancy counter to
CPU before counting pairwise overlaps. The original fitting, ranking, coefficients,
data, task splits, thresholds and evaluation code are unchanged. A CPU regression
test and a live-reservation CUDA-to-CPU counting probe precede the corrected fit.

Use `experiments.structure_net.jspace_bench_tensor_v2` with new artifact root
`data/experiments/jspace_bench_tensor/20261113_v2`. It content-hashes the added
wrapper, this amendment, and its regression test alongside all unchanged producers.
It references the completed v1 shakedown/capture through verified manifests with
explicit original fingerprints and `execution_reused=true`; neither is rerun or
reclassified. The previous failure remains one recorded systems failure in the
campaign. The original prospective scientific protocol and decision rules apply.
