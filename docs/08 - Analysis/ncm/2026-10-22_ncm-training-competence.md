# NCM training competence after adapting the paper's regime

Date: 2026-10-22. Two separate exploratory, one-seed training diagnostics.
Both local competence gates passed. **Neither establishes95% held-out summary
accuracy.** The256-case final test remains sealed; no speedup is measured.

| Measurement | Result | Interpretation |
| --- | --- | --- |
| Free code accuracy on32training pairs |6,379/6,400=99.6719%|Training memorization only|
| Coarse codes with matched sources |128/128|Training conditioning control|
| Coarse codes with cyclically shifted sources |1/128|Source-sensitive generation|
| Decoded target tokens, original decoder |1,132/1,149=98.5205%|Codec errors persist despite learned codes|
| Exact training texts, original decoder |19/32|Not task accuracy|
| Full-data original decoder reconstruction |68,431/69,682=98.2047%|Fixed baseline|
| Full-data polished decoder reconstruction |69,515/69,682=99.7603%|Training reconstruction only|
| Exact texts after decoder polish |1,856/1,999=92.8464%|Training reconstruction only|

The paper-style adaptation first trained the actual tokenizer encoder,
resampling projections, EMA codebooks and decoder with nested-cutoff losses.
The frozen-codec generator then received unpaired training-source windows,
bridge warmup and conditional source/summary pairs. The completed larger
noisy-pair run still produced invented names and malformed wording; its first
complete development judgment is0/64under Sonnet, with the remaining independent
model/judge results being recorded separately. Low token CE was not accepted
as evidence of task success.

The coarse follow-up continued the native generator on32fixed training pairs.
It preserved the native source embedding channel, added a zero-initialized
global residual from the frozen Llama final valid hidden state, enabled the
released implementation's optional autoregression over the first4codes, and
used coarse-only warmup followed by equal total loss per hierarchy level and
conditioning corruption. The original tokenwise objective gave the first4of200
positions only2%of loss weight. This is one bundled intervention, so the study
does not isolate which individual change caused improvement. Shuffling both
native and body source inputs shows source dependence; it does not isolate the
Llama body's contribution. All original body layers, including their MLPs,
remained frozen. Checkpoint reload and causal coarse-input checks passed.

The decoder follow-up froze every encoder, codebook, residual and resampling
state, then trained only full-cutoff text reconstruction. Every non-decoder
state byte and every one of the1,999summary code assignments stayed unchanged.
This reduces word corruption while retaining the discrete language already
learned by generators. Pairing with a new decoder requires an explicit
compatibility receipt; original checkpoint metadata is preserved.

The original tokenizer, generator and new source interface use released
checkpoint warm starts; the source residual projection starts at zero.
These short adaptations are far smaller than the paper's training budget.
The decoder phase is an engineering addition rather than a claimed reproduction
of the published recipe. [NCM paper](https://arxiv.org/html/2610.14162v1).

Measured training compute was289.0seconds for the32-case curriculum and
59.1seconds for decoder training plus recorded checks, on a sharedRTX3060.
These are not inference benchmarks. Peaked own allocation was3.289GB and1.652GB
respectively. The prototype supports512GPT2source tokens and48target tokens,
which does not cover the full application range. Known teacher-label noise
(30/32audit) and one-seed training-fit limitations remain disclosed.

A separate [full-data continuation](../07%20-%20Status%20Reports/2026-10-22_ncm-full-curriculum-plan.md)
has passed its GPU lifecycle pilot and is running. It must earn its own64-case
development accuracy under both qualified judges. No result from that run is
included here. The user's95% complete-summary objective remains open.

Aggregate: `data/experiments/ncm_paper_regime/20261022_competence_results_v1/campaign_results.json`;
SHA256 `1162b8bdf6a8c54be8d26758511edd1ec13fb4b95758881ba6d222f2880a3a5d`.
Reproduce with `train_ncm_coarse_curriculum` and `train_ncm_decoder_polish`
under `experiments.structure_net`, using the pinned moba environment and GPU0.
The immutable plans, producer source copies, hashes, raw predictions, code
labels, optimizer/RNG checkpoints and memory records remain in the run roots.
Registry records keep confirmed=false and underpowered=true.
