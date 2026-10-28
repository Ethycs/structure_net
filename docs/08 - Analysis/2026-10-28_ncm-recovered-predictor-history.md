# Recovered predictor learns much more from correct code history than from rollout

Exploratory study `ncm-recovered-predictor-history-diagnostic-v1` completed with
all execution-consistency controls passing. On the four preselected recovered
TRAIN examples, correct-history code agreement is 57.33 percent, while source-only
rollout agreement is 6.08 percent. These are code diagnostics, not semantic summary
accuracy; the 95 percent application goal remains unachieved.

The result is
`data/experiments/ncm_paper_regime/20261028_recovered_predictor_history_diagnostic_v1/result.json`,
SHA256 `f284cca9421e19ddc1e3efd4c2cb307b983da93758961f19907a674f73f0dd1e`.
Actual tool session 41609 ended at chunk a64559, exit 0. The receipt SHA256 is
`039a3f52f33e54550afa159e5facd6201e1260ef921d067bc32ab40200fbde3d`.
The producing code, tests and prospective plan are commit c53ea46.

| Condition | Matching reference codes / 2564 | Agreement | Mean level CE |
|---|---:|---:|---:|
| Correct earlier codes, matched source |1470|57.33%|3.554|
| Own generated earlier codes, matched source |156|6.08%|11.999|
| Random earlier codes, matched source |60|2.34%|7.476|
| Correct earlier codes, cyclically swapped source |1458|56.86%|3.870|

Every explicit source-only rollout exactly reproduced the saved step 6416 codes.
Packed predictions conditioned on those generated histories reproduced explicit
generation at all 641 positions in all four cases. Root logits were invariant to
correct, generated or random later history within the declared 1e-4 tolerances,
with identical argmax. The maximum root discrepancy was 1.1444e-5. All model and
protected tensor hashes stayed unchanged. There was no training or gradient use.

The four reference root codes were never selected. Their ranks under the
source-only root distributions were 305, 41, 2 and 24, in original control order.
Across the first six levels, even correct-history predictions matched only
132/484 reference codes. At the final level they matched 437/512. Thus the overall
teacher score conceals substantially weaker coarse-level prediction. Giving
correct history does not establish that the source-to-code task is well learned.

Changing the source while retaining correct history barely changes the total
code-match count. It does increase CE, and changes root argmax on three cases;
the predictor is not wholly source-independent. The evidence suggests heavy
reliance on reference-code history and weak coarse prediction, followed by a
large rollout degradation. The lower CE under random histories than under its
own histories is also consistent with badly calibrated predictions on the
model's actual rollout states. These are descriptive observations on four
nonrandom TRAIN controls, not a representative estimate or proof of a remedy.

The prior fixed-code readout experiment reconstructed all four references
exactly using the improved standalone decoder, but could not rescue these same
source-predicted codes. Together these findings place a bottleneck in source-to-
code prediction for these examples. They do not imply that code agreement must
be exact for a valid paraphrase, or that the codec is universally lossless.

An independent readback recomputed every per-level match count from the saved
codes and every root-consistency comparison from saved raw logits. It also
derived the root ranks above. Its root is
`20261028_recovered_predictor_history_replay_v1`, actual session 88026,
chunk 795056, exit 0, receipt SHA256
`e4b2773b8abe4d60b02a17a7ee778babadc5d4d3ed4592ace58d6ea68a365262`.
It did not rerun the model or independently recompute full-hierarchy CE logits.

Execution used locked Pixi on physical GPU1 RTX2060, FP32, TF32 off, batch 1,
complete original source features and the audited final 6416-update checkpoint.
Peak GPU allocation was 1,277,061,632 bytes, below the 4.9 GB limit. Three focused CPU
tests passed before launch. No held-out source, semantic judge or Final320 was
used, and no model throughput was measured.

The next decisions must incorporate the pending source-disjoint candidate DEV
results. A generic repeat of generated-history training is not already justified:
the older matched fixed-target/residual-target study trained on generated history
and both arms remained 0/128 on semantic DEV. That earlier study used only 1,376
pairs, so it does not settle whether the larger recovered corpus or stronger
source conditioning could help. Preserve that failure when designing any new
curriculum; target both coarse source learning and inference-history robustness.

Separately, the expanded candidate's two-case inference pilot completed with
exact repetition and 1,254,749,184 bytes peak allocation. Full 128-case generation
has been launched through the qualified process-local hash-cache wrapper at
commit 65f8189. That change avoids repeated verification reads inside one process;
it changes no model computation, data, gates or semantic judgments.
