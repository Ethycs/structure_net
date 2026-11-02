# Upstream repositories and the changes our adaptation needs

Date: 2026-10-23. Read-only implementation audit, not a local speed benchmark
or a95%accuracy result. Pinned raw evidence:
`data/experiments/ncm_paper_regime/20261023_upstream_repository_comparison_v1`.

The most useful training reference is
[SGLang SpecForge](https://github.com/sgl-project/SpecForge/tree/f586b1ee40b5e4bf3ef03954849024b57a5edb7a).
It contains actual DFlash/EAGLE training implementations and a
[Qwen3-8B DFlash configuration](https://github.com/sgl-project/SpecForge/blob/f586b1ee40b5e4bf3ef03954849024b57a5edb7a/examples/configs/offline/colocated/qwen3-8b-dflash.yaml).
We inspected its source; we have not trained or benchmarked SpecForge locally.

| Repository | What it provides | What it establishes for this project |
| --- | --- | --- |
| [HazyResearch/ncm](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) | Official NCM model, released checkpoints and evaluation configuration | Our vendored README, XSum config and model source match the pinned release. The inspected47-entry tree contains no training loop; our custom training is not a reproduction of an author-provided trainer. |
| [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge/tree/f586b1ee40b5e4bf3ef03954849024b57a5edb7a) | Concrete native-vocabulary draft-model training, target-feature providers and Qwen recipes | Strong reference for implementing and testing the teacher/student interface. Its speculative decoding keeps the full parent as verifier. That mechanism cannot establish standalone NCM quality. |
| [z-lab/dflash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | DFlash implementation, model/usage information and training integration | Useful parallel-token drafting reference. Reported upstream behavior is not a measured speedup on our hardware. |

The original local adaptation projected one final Llama vector into a GPT2/XSum
decoder prefix. It changed the released coarse generation into four AR coarse
codes followed by parallel refinement. This introduced both an unvalidated body
interface and a task/vocabulary transfer problem. High training code or token
recovery did not demonstrate a working summarizer.

The inspected SpecForge DFlash path conditions on per-token features from several
teacher layers and reuses native embeddings and the native word head. Those are
concrete interface choices worth copying. We implemented full per-token native
features and native-vocabulary controls. However, the five-block borrowed native
readout memorized32/32training texts and then scored0/64development summaries,
including0/64readable. Its planned full NCM capacity run was therefore explicitly
deferred. Retaining a vocabulary and selected blocks did not retain usable
language in that configuration; this does not isolate pruning as the sole cause.

The intact16-block1B native language control scored50/64whole-summary passes,
53/64faithful,61/64salient and64/64readable. This is an AR baseline, not an NCM
success. These counts use the original approximately32-word salience rubric;
a later explicit128-token evaluation amendment preserves all old scores and
requires fresh judge qualification before application-budget comparisons.

What we should do next follows from those differences:

1. Preserve an intact pretrained language path and test an exact no-op interface
   before training a new conditional component. Frozen native MLP weights are
   compatible with training new heads/adapters; a learned input bridge should
   not silently replace all pretrained language conditioning.
2. Qualify actual training summaries under the application's128-token budget.
   The old113/128teacher audit and the new8/10pilot diagnostic do not support
   assuming the teacher labels meet95%. Fix data production before scaling.
3. Train and measure actual generated histories, including noisy/wrong coarse
   histories, using correctly conditional residual targets. Training fit and
   target-conditioned repair probes remain mechanism controls only.
4. Require complete-summary faithfulness, salience, readability and token-cap
   passes on source-disjoint data before speed comparisons. Keep final cases
   sealed until development gates pass and represent the full input range.
5. If adopting DFlash's full-parent verification, report that as a separate
   speculative system with its full compute/memory costs. Do not call its
   verifier-protected accuracy evidence for a standalone NCM replacement.

The official released NCM was previously checked against our local FP32
inference on32XSum examples, with32/32exact decoded texts. That is a limited
inference implementation control, not a training replication or95%summary
quality result. No repository inspection fixes the failed custom model by
itself. The relevant source snapshots, hashes, first failures and negative
judgments remain append-only.
