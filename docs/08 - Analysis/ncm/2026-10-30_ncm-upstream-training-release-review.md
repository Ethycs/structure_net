# NCM upstream training release check

October 30, 2026; checked at approximately 22:42 UTC. Read-only investigation;
no training, generation, grading or service changes. This follows the
[failure review](2026-10-30_ncm-failure-literature-and-code-review.md) and
[untouched-checkpoint tests](2026-10-30_ncm-pretrained-capabilities.md).

**The official repository is available, but its full training loop has still
not been released.** Current `main` is
`c70274273f489abc73832b2b98584a8483b6b203`, dated October 14. All **39 files**
in its complete Git tree match our existing `ncm-20261021` vendor byte for byte,
including the configs and assets. There is no new author implementation to
import. The five-commit history contains the original release, README/citation
changes and evaluation configs. The branches endpoint lists only `main`; the
releases endpoint is empty. These are observations at this check, not a claim
about future releases. [Pinned repository](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203),
[latest commit](https://github.com/HazyResearch/ncm/commit/c70274273f489abc73832b2b98584a8483b6b203).

The released entry point is `evaluate.py`, with `config/eval/` and generation /
scoring scripts. No author optimizer loop, training entry point or complete
executable training configuration is present. Training-code requests
[#2](https://github.com/HazyResearch/ncm/issues/2) and
[#10](https://github.com/HazyResearch/ncm/issues/10) remain open. The author's
affirmative reply in #10 promises intent to add it; it does not supply code.
Recent community proposals, including
[#7](https://github.com/HazyResearch/ncm/pull/7) and
[#18](https://github.com/HazyResearch/ncm/pull/18), are unmerged contributions,
not an author training release. **There is no newly available official
training command to run.**

**A concrete additional finding: the evaluation YAML does not execute the
complete codec objective it describes.** The
[LM1B config, lines 111–124](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/lm1b-small.yaml#L111-L124)
specifies `fine_drop.dual: true`, full/truncated weights `0.7/0.3`, and
`aux.code_corruption`. The [paper's tokenizer table](https://arxiv.org/html/2610.14162v1#A3.SS1)
also describes the full/truncated mixture. However:

- The released quantizer reads the dropout probability, minimum retained level
  and optional annealing settings. No released Python source reads `dual`,
  `w_full`, `w_trunc` or `code_corruption`.
- In training mode, `VQVAE.forward` decodes one quantizer result. With LM1B's
  `fine_drop.prob: 1.0`, that result is always truncated before the final level;
  this call alone cannot implement the stated 70/30 full/truncated objective.
- The mixed reconstruction losses and configured corruption consequently
  require additional trainer logic that is absent from this release. The
  precise implementation cannot be recovered from those config fields alone.

These statements follow the actual
[quantizer configuration reads](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/quant.py#L263-L268),
[dropout selection](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/quant.py#L448-L532)
and [VQVAE forward](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/models/vqvae.py#L91-L95).
This is a reproducibility gap, **not evidence that our latest run accidentally
trained only truncated reconstructions**: its codec was frozen, and the prior
[decoder adaptation](../../../experiments/structure_net/ncm_recovered_decoder_training.py)
explicitly decoded full codes with its own loss. Our older codec trainer also
uses a custom cutoff objective. Any future codec replication must specify the
external mixture and corruption explicitly instead of assuming YAML parity
establishes training parity.

**The most useful author training observation remains the reconstruction /
generation tradeoff.** In [issue 16](https://github.com/HazyResearch/ncm/issues/16#issuecomment-6470658660),
the author reports that corrected dead-code restarts improved reconstruction
and code utilization but worsened downstream generation metrics. This is an
author-reported experiment, not a local reproduction. It supports selecting a
codec jointly for reconstruction and predictability; maximizing reconstruction
alone is insufficient.

The same issue contains a community report of tied-codebook logit-scale
problems. Neither it nor the restart bug explains the latest local run:
LM1B uses a shared linear output head, and frozen codebooks perform no EMA
restarts. The old root-gradient problem is already fixed in the active
[teacher-input builder](../../../experiments/structure_net/ncm_trainable_example_batching.py).
This audit establishes no new applicable defect in that training path.

The comparison to our completed adaptation remains:

| Training property | Released recipe / implementation | Our latest model and implication |
| --- | --- | --- |
| Task and source interface | XSum has a 512-position source prefix and 48-position output. LM1B has a one-position null prefix and 128-position output. | We attached the unconditional LM1B generator to frozen Llama features through three late source-attention modules. This conditional interface had to be learned; it was not inherited from the news summarizer. |
| Training stages | Train a codec, freeze it, then train the hierarchy predictor. | We already trained the generator with the codec frozen. Sequential training is not a newly missing step. |
| Noisy conditioning | Earlier-code corruption with clean targets; equal-level loss is discussed in the paper. | Equal-level loss and uniform 25%-to-5% corruption on half the presentations were already used. Generated-history and residual-retargeting variants also already failed their semantic comparison. |
| Conditional training scale | Paper XSum: 350,000 updates, batch 512, peak LR 3e-4 after unpaired pretraining. | Latest stage: 6,416 updates, batch 32, peak LR 3e-5; 43,894 main pairs plus 2,384 anchors, training 105.3M parameters. Approximately 873 times fewer sequence presentations in this stage; not a total-compute ratio or required-budget estimate. |

Sources: pinned [XSum](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/xsum.yaml)
and [LM1B](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/lm1b-small.yaml)
configs; [paper Appendix C](https://arxiv.org/html/2610.14162v1#A3) and
[corruption ablation](https://arxiv.org/html/2610.14162v1#A6.SS3);
local implementation and run records linked in the prior failure review.

**What this changes:** copying upstream again will not repair our adaptation.
The new static finding explains why the released model classes and YAML are
insufficient for faithful training replication. It does not overturn the
observed failure: our four fixed TRAIN controls fall from 57.33% reference-code
agreement with correct history to 6.08% with generated history, while the
completed semantic result is 0/128. The untouched Sudoku models' 99.4% easy /
86.3% hard scores demonstrate real specialist competence; they do not establish
faithful conversational summarization or successful training reproduction.

If this research resumes, the useful prerequisite is a native conditional
baseline that passes the intended semantic rubric, followed by a controlled
comparison of its native source prefix against the proposed Llama interface.
Measure source sensitivity and generated-history quality as well as codec
reconstruction. Obtaining the author's complete codec and generator trainers
would permit an actual recipe comparison, especially the missing LM1B loss
mixture. None of this is a demonstrated repair, and another unrestricted
training run is not justified by the repository's present release status.

Evidence is preserved in
[the audit directory](../../../data/experiments/ncm_paper_regime/20261030_upstream_training_release_review_v1):
complete Git tree, commit/branch/release responses, issue snapshots and author
comments. [Vendor parity](../../../data/experiments/ncm_paper_regime/20261030_upstream_training_release_review_v1/vendor_parity.json)
recomputes Git blob hashes for every released file; all 39 match.
The manifest binds the evidence and this report by SHA-256. Verification used
locked Pixi with CUDA disabled. No model weights were loaded.
