# Batching four examples nearly halves the measured NCM training component time

The forward/backward component ran1.90times faster with four examples per
microbatch on the RTX2060, using3.674GB peak allocation. Preserving frozen codec
conditioning preparation separately for each example resolved the numerical
discrepancy found in the first two comparisons. The unchanged strict FP32
gradient gate passed for a batch of two, with clean and corrupted histories.
This is a systems result; no optimizer step, semantic evaluation or application
inference speed test ran. Complete-summary task accuracy remains unqualified.

Study `ncm-example-batching-v1` is classified
`original_batching_parity_failed_serial_codec_passed`. The independently replayed
aggregate is
`data/experiments/ncm_paper_regime/20261026_example_batching_results_v1/campaign_results.json`,
SHA256 `52aa11ce9c2312de269061627cac8117ecfa1e6df03529776f056913374cc6d6`.

The [original plan](../07%20-%20Status%20Reports/2026-10-26_ncm-example-batching-plan.md)
compared separate examples with one padded batch containing the shortest and
longest existing TRAIN sources:197 and2273 complete source positions. Both
executions used identical target codes and conditioning noise. All105,323,136
trainable generator/source-adapter parameters were included. The codec and word
adapters stayed frozen, dropout was disabled for parity, and TF32 was off.

| Execution variant | Clean relative gradient L2 error | Noisy relative gradient L2 error | Fixed parity gate |
| --- | ---: | ---: | --- |
| Fully batched preparation, default attention | 8.3593e-5 | Not run after failure | Failed |
| Fully batched preparation, MATH attention | 8.3553e-5 | Not run after failure | Failed |
| Per-example codec preparation, batched generator | 3.2867e-6 | 2.8133e-6 | Both passed |

The first two variants exceeded the5e-5 relative gradient limit and failed
logit allclose(rtol=atol=1e-4), despite zero argmax differences. Their failed
gates remain failed. The
[attention-backend control](../07%20-%20Status%20Reports/2026-10-26_ncm-example-batching-math-control.md)
did not resolve the discrepancy.

The subsequent
[codec-preparation control](../07%20-%20Status%20Reports/2026-10-26_ncm-example-batching-codec-control.md)
measured486,789 different values in the2x640x384 conditioning tensor when the
same frozen VQ operations ran separately versus batched. Maximum difference was
0.0002061 and relative L2 difference2.1375e-5. Keeping that small frozen
calculation per example, then concatenating before the generator, reduced the
observed discrepancy enough to pass every unchanged numerical requirement.
Maximum logit differences were4.9591e-5 clean and5.3406e-5 noisy; all1282 code
argmaxes matched in each comparison. This localizes the dominant discrepancy
to the preparation execution in these cases. It does not prove universal
bitwise equivalence or identify a particular low-level convolution algorithm.

After those checks passed, the resource sweep used the eight longest distinct
TRAIN sources,2191–2273 positions, without clipping. The effective batch stayed
eight. FP32 weights, FP16 autocast, FP32 cross-entropy, fixed loss scaling128,
active dropout and the same example-ordered conditioning noise were used at
each size. Each passing size had one warmup and three measured effective batches.

| Microbatch size | Median seconds per8 examples | Relative throughput | Peak allocated GPU bytes |
| --- | ---: | ---: | ---: |
| 1 | 1.5440 | 1.00x | 2,203,734,016 |
| 2 | 0.9703 | 1.59x | 2,687,308,288 |
| 4 | 0.8131 | 1.90x | 3,673,966,592 |
| 8 | No admitted timed trials | Not reported | 5,239,772,672 |

Batch8 completed its warmup but exceeded the fixed4,900,000,000-byte allocation
budget, so the sweep stopped. It did not run out of physical GPU memory. Its
single warmup timing is retained in raw evidence and is not a throughput result.
All completed batches had finite losses and gradients. Weights were unchanged
under the full checkpoint tensor hash after each of the three diagnostics.

Timing includes collation, transfer, conditioning noise, forward/backward and
gradient-finiteness checks on already-loaded features. It excludes original
Llama body extraction, disk reads and AdamW work. It therefore establishes
neither overall training throughput nor application token throughput. Different
dropout draws and FP16 arithmetic across shapes preclude identical training
trajectories. The real-model FP32 gradient check covers batch2; batch4 currently
has resource qualification only. Before adopting batch4, qualify its gradients
and a complete optimizer/save/reload lifecycle in the larger training study.

Twelve focused CPU tests passed across collation, source masking, noise-order
invariance, uneven microbatch weighting, original objective/source-attention
contracts, and aggregate gate arithmetic. A separate CPU process replayed input
selection, source and checkpoint bindings, frozen source snapshots, per-parameter
gradient reduction arithmetic, ordered stop rules and timing medians. It did not
repeat the GPU forwards or independently recompute gradients from raw tensors.
Actual terminal receipts for all runs and the audit are retained. Both earlier
failures are included in the aggregate and conservative registry record.

The next substantive work remains the larger paired-data conditional training
study. Public codec diagnostic sources must be excluded from its TRAIN split,
and its data/feature-storage plan and batch4 optimizer lifecycle still need to
be completed. This batching result does not establish the95% semantic goal;
the latest completed application evaluation remains0/128 passes per matched
arm, and all sealed final sources remain closed.
