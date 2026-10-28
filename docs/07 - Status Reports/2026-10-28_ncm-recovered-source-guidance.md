# Fixed-weight source guidance on recovered TRAIN controls

Exploratory hypothesis `ncm-recovered-source-guidance-TRAIN-v1`: amplifying the
source-conditioned logit contribution can improve the latest predictor's poor
free generation without fitting weights. Prior evidence is the completed
step6416 history diagnostic: 1470/2564 reference codes with correct history,
156/2564 on rollout, 1458/2564 with correct history and swapped source. These four
controls were selected before training, are nonrandom TRAIN examples, and cannot
qualify the application accuracy goal.

The [NCM paper, Appendix F.4](https://arxiv.org/html/2610.14162v1#A6.SS4)
reports per-scale classifier-free guidance, with XSum ROUGE peaking at weight 0.5.
The pinned author implementation `models/next_concept.py::_sample_level` ramps
the effective weight from zero at the root to the configured weight at the final
level. The root therefore receives no guidance in that implementation.

Use the same four complete sources, latest audited predictor, fixed improved
step6212 standalone decoder, and unchanged native renderer as the completed
readout diagnostic. All predictions receive source features only. Reference
codes and texts are used for post-generation diagnostics, never to choose codes.
No model selection, training, semantic requests, DEV or Final320 access occurs.

Run four fixed arms in this order for every case:

1. Baseline: zero guidance, ordinary greedy next-scale generation.
2. Author ramp: scale 0.5, effective weight `0.5*level/11`.
3. Constant 0.5: effective weight 0.5 at all levels, including the root.
4. Constant 1.0: effective weight 1.0 at all levels.

For nonzero effective weight `t`, combine `(1+t)*conditional-t*unconditioned`
logits before argmax. Both forwards use the same generated cumulative history.
The unconditioned forward bypasses the added source-attention adapters while
retaining the native CLS prefix, positions, generator and code history. Restore
the source context after every call, including exceptions. This is a local
adaptation of guidance, not an exact replication of the authors' source-prefix
null branch. Our model has not been trained with a dedicated null-source
condition; null-branch distribution shift is a limitation, and failure here
would not invalidate classifier-free guidance in the paper.

All sixteen code outputs must repeat exactly. Baseline codes and decoded words
must match the saved latest-predictor/improved-codec control outputs exactly.
The author-ramp root must equal baseline. Verify complete finite word logits,
unchanged model and codec hashes, and restored source context. Preserve any
failure and stop; no automatic retry or parameter sweep outside these four arms.
Report every raw code/word output, text, per-arm code matches and model-call
counts. Code matches are mechanism diagnostics, not task-accuracy estimates.
Qualitative comparison may motivate a later prospectively defined training or
development study; it cannot qualify this candidate by itself.

Run the three focused CPU contracts before launching this bounded inference
pilot. Locked Pixi, physical GPU1 RTX2060, FP32, TF32 off, batch 1, deterministic
algorithms, seed 42, no gradients, peak own allocation limit 3.5 GB. The unchanged
features include the 3464-position longest control. No other job may share GPU1.
The inherited stat-guarded hash cache reads each input fully on first access;
four selected feature payloads are hash-verified independently. Bind source,
plan, tests, qualification, environment and inputs into a fresh producer.

Root `20261028_recovered_source_guidance_TRAIN_v1`; entry point
`experiments.structure_net.ncm_recovered_source_guidance`. Existing semantic
candidate ordering and all first judgments remain unchanged. Guidance adds
generator calls; no speedup is claimed or measured in this diagnostic.
