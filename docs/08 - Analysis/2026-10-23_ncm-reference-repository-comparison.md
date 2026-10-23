# NCM reference repositories and changes needed in our adaptation

Date: 2026-10-23. Classification: source audit with descriptive scoring of
existing public benchmark outputs. Evidence root:
`data/experiments/ncm_paper_regime/20261023_repository_training_audit_v2`.

The official NCM checkpoint provides a runnable reference for parallel concept
generation. Our current frozen Llama hybrid changes its codec, attention,
decoder and training distribution substantially. Its failure does not establish
a failed reproduction of the published architecture. For training an existing
causal Llama to decode faster, Consistency LLM and JacobiForcing provide actual
training pipelines. Those methods should be evaluated separately from NCM.

| Reference | What is available | Evidence and practical boundary |
| --- | --- | --- |
| [Official NCM](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) | Learned multiscale codec, concept generator, checkpoints and evaluation scripts | The existing local implementation and author script agree on all 32 saved XSum outputs. The pinned release has 39 files and no training entry point. |
| [Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM/tree/22775363f8563c63620e71f7a204e90a51d6a379) | Llama implementation, trajectory collection, clean and noisy training, checkpoints and evaluation | Useful executable training reference for the Llama family. Published models are 7B; this audit does not reproduce their training or our frozen 3B adaptation. |
| [JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing/tree/028dbd6283fdc6fa356f0b109ce04c5ca538e49f) | Trajectory collection and packing, paired training losses, parallel inference and correctness tests | A newer training reference for causal parallel decoding. The example launch uses Qwen code/math models, four GPUs and full tuning; its published performance does not establish attention-adapter-only summarization results. |

NCM issue [2](https://github.com/HazyResearch/ncm/issues/2), requesting the
Sudoku training code, remains open without comments at audit time. Four
candidate fork branches and the training-related architecture PR
[7](https://github.com/HazyResearch/ncm/pull/7) were also inspected. Their
snapshots do not provide a verified complete NCM training replication. The
softwarewrighter fork reports Sudoku inference on Apple hardware; its report
explicitly leaves XSum evaluation unfinished. This search is not an assertion
about every public fork.

The new descriptive score of the **same 32 stored released-checkpoint
outputs** is ROUGE-1 **34.6064**, ROUGE-2 **12.9796**, ROUGE-L **27.8921**.
The paper reports **35.0 / 12.8 / 28.0** on the full XSum benchmark.
These different denominators preclude a full benchmark replication claim.
ROUGE is reference overlap, not the application's complete-summary pass rate.
The paper's speed comparisons also do not establish a speedup over our Llama
baseline. [NCM paper, Table 2](https://arxiv.org/html/2610.14162v1#S4.T2)

## Architectural differences that affect the experiment

The released XSum configuration uses a 512-token GPT-2 source and a 48-token
target. The application requires sources up to 2,048 cl100k tokens and stored
summaries up to 128 cl100k tokens. Changing tokenizer or frame length requires
new training and evaluation; the existing checkpoint does not cover that range.
[Released XSum configuration](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/xsum.yaml)

| Component | Released NCM XSum | Latest trained native hybrid |
| --- | --- | --- |
| Text representation | Learned VQ encoder, quantizer and text decoder trained for reconstruction | Residual clustering over fixed native Llama token embeddings; no learned codec decoder |
| Hierarchy | Nine released levels, 4/6/10/14/20/26/32/40/48, with 4,096 entries per level | Five levels, 4/8/16/32/64, with 256 entries per level for a 128-token word frame |
| Finest representation | Includes a code level at the full target length | Ends at half the word-frame length before the native word stage |
| Within-level computation | Parallel concept queries with the configured block mask | Causal queries through the original 28-layer Llama body with new rank-eight attention adapters |
| Text production | Dedicated decoder reads the completed latent reconstruction in parallel | Native causal LM head reads predecessor-word estimates; originally four word refinements |
| Codec learning | Reconstruction and quantization objectives; fine-level dropout | K-means distance objective on frozen embeddings |

These comparisons follow the pinned
[VQ model](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vqvae.py),
[quantizer](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/quant.py),
[text decoder](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vae.py),
and [generator](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/next_concept.py),
against local `ncm_native_concepts.py`, `ncm_native_query_concepts.py` and
`ncm_native_word_neighborhood.py`. The paper's XSum row reports eight generator
evaluations; the released configuration has nine levels. Preserve that
difference in a reproduction report.

The local diagnostic isolates a major practical problem: on four training
summaries, correct target-derived concepts plus four word refinements recover
only 37/346 words, versus 332/346 with correct previous words. The causal word
stage still depends heavily on a correct word history. That is not how the
released NCM text decoder receives its conditioning. Extending decoding to a
verified fixed point produced coherent text but only **41/64 complete-summary
passes**, versus **52/64** for unchanged 3B. The original nine-pass result
remains **0/64**. These observations do not isolate the codec as the sole cause.
[Word mechanism result](2026-10-23_ncm-word-mechanism-and-upstream-actions.md),
[complete-summary evaluation](2026-10-23_ncm-converged-native-development-quality.md)

## What to take from the causal training repositories

Both training references collect intermediate Jacobi states and converged
states from their model. Their clean and noisy branches are paired: the clean
branch supplies hard autoregressive CE weighted by ten and detached soft
targets for the noisy branch. The JacobiForcing multiblock trainer also handles
divergence, EOS, padding and block alignment explicitly.
[Llama trainer](https://github.com/hao-ai-lab/Consistency_LLM/blob/22775363f8563c63620e71f7a204e90a51d6a379/cllm/cllm_trainer_global.py),
[JacobiForcing trainer](https://github.com/hao-ai-lab/JacobiForcing/blob/028dbd6283fdc6fa356f0b109ce04c5ca538e49f/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py)

Our original hybrid alternated clean/noisy losses and used external summary
wording plus offline original-teacher top-32 distributions. The later paired
control did implement clean CE times ten plus detached soft consistency, but
used only four training cases, the already failed shared adapter bank, and up
to four-step states. Those states were not demonstrated to converge to the
external summary labels. That control failed its token-recovery gate. Adding
the same two losses again would not address this remaining mismatch.

For a causal acceleration experiment, first qualify a clean summary model,
then collect that exact checkpoint's own converged outputs and intermediate
states on training sources. Preserve a stable clean path while training faster
updates. Use the Llama reference for model alignment and the newer repository
for trajectory packing and decoding. An attention-only adapter version remains
a new experiment under the frozen-body constraint.

The live 512-update isolated word-bank run is the clean quality control for
this sequence. It deliberately omits concept hints, noisy training and codec
losses. Its current training loss cannot establish NCM quality or acceleration;
complete-summary evaluation must follow its fixed final checkpoint.

## What the attached NCM experiment should do

For the requested NCM module attached to a frozen transformer, preserve the
mechanisms that make NCM parallel:

1. Establish a learned native-token codec and parallel decoder that reconstruct
   complete unseen summaries through a full-length finest code level. Measure
   reconstruction quality, entities and numbers before predicting any codes.
2. Freeze that qualified codec. Train a conditional concept generator from the
   parent's per-token source representations to its code hierarchy, using the
   NCM level mask and earlier-level reconstruction as conditioning. This
   attachment is an adaptation and needs its own controls.
3. Train the new modules, including their new feed-forward weights. Keep all
   original parent weights, including its MLPs, frozen. The paper does not show
   that its codec and decoder can be replaced by a frozen native causal head.
4. Evaluate source-only complete summaries on fresh sources across the full
   application range. Require the task-quality gate before comparing total
   prefill and decode latency, memory and tokens per second on matched hardware.

NCM and causal trajectory distillation answer different questions. A successful
clean teacher admits the second experiment; it does not establish that an NCM
replacement works. Neither reference currently demonstrates our 95% final
application target.

## Evidence and reproduction

`manifest.json` records the three upstream commit identities and repository
trees. `source_manifest.json` and `supplement_source_manifest.json` bind every
downloaded source file to a URL and SHA-256. `candidate_forks.json` and
`ncm_pr7_tree.json` preserve the inspected fork and PR trees.
`released32_descriptive_rouge.json` records all per-example scores and the
unchanged input artifact hash
`91dd721295e1eb18db101ade81ecb2c308754817fb84d60ddc7a1e4170d3f769`.
Scoring uses `rouge_score` 0.1.2 with stemming, F1 multiplied by 100, averaged
over all 32 cases. No new model generation or judge request was used for that
descriptive comparison. `comparison.json` binds the audit to local sources and
the existing quality results. Completed training artifacts remain unchanged.
