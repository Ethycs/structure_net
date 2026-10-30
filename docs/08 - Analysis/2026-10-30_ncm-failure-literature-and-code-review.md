# Why the NCM adaptation is failing: literature, code and experiment review

Research date: October 30, 2026, America/Los_Angeles. Training and remaining
grading stay stopped. This review makes no model or judge calls and opens no
sealed final cases. It includes a new descriptive analysis of existing TRAIN
metadata, code labels and optimizer traces.

**The strongest explanation is weak source-conditioned code prediction,
amplified by a large mismatch between training histories and generated
histories.** Our latest implementation adapts an unconditional LM1B generator
through a new Llama interface. We have not established that this interface
transfers summarization competence. Better reconstruction cannot supply the
source facts that the predictor failed to select. This is a supported diagnosis
of the observed failure, not a proven single cause or a successful repair.

The completed latest semantic result remains **0/128**, with 128 valid Terra
judgments and an independent replay. The stopped alternative readout has
0/113 judged cases and is incomplete. These are whole-summary semantic checks,
accepting paraphrases, rather than exact-reference scoring. See the
[completed evaluation](2026-10-29_ncm-latest-recovered-semantic-grade.md).

**1. The predictor learns much more from the supplied answer history than from
the source.**

The [latest checkpoint diagnostic](2026-10-28_ncm-recovered-predictor-history.md)
uses the same four preselected TRAIN examples and final step6416 checkpoint:

| Diagnostic | Measured result | What it establishes |
| --- | ---: | --- |
| Correct earlier codes, correct source | 57.33% reference-code agreement | Partial prediction competence with privileged history |
| Correct earlier codes, swapped source | 56.86% | This code-match measure barely responds to source substitution |
| Own generated history, correct source | 6.08% | Severe degradation during actual generation |
| Random history, correct source | 2.34% | Random-history agreement is also very low |
| Correct-history mean level CE | 3.554 | A training-like diagnostic |
| Own-history mean level CE | 11.999 | Actual rollout states are especially difficult |
| Random-history mean level CE | 7.476 | Uniform noise does not recreate the measured rollout distribution |
| Correct first/root code | 0/4 | Weak source-only coarse prediction on these controls |

Correct-history agreement in the first six levels is only 132/484 (27.27%);
the final level achieves 437/512 (85.35%). An overall code score therefore hides
where prediction is weakest. Source substitution changes root argmax on 3/4
cases, so the source is not wholly ignored. The conclusion is weak useful
conditioning, not complete disconnection.

The improved decoder reconstructs these four references exactly when supplied
their correct codes. That locates a bottleneck before the decoder on these
examples. It does not prove universal codec sufficiency or imply that valid
paraphrases require identical codes.

Our new trace analysis also confirms actual optimization: mean logged training
CE falls from 3.018 to 2.740, 2.640 and 2.591 across the four epochs. Those are
mixed clean/noisy training losses, not held-out or rollout losses. Updates were
learning the training objective while application quality remained unusable.

**2. Our conditioning route changes a central part of the released model.**

The [released conditional XSum configuration](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/xsum.yaml)
has a 512-position source prefix and a 48-position output canvas. Our current
base is the [unconditional LM1B configuration](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/lm1b-small.yaml):
one CLS prefix, 128 output positions, twelve levels and 8,192 entries per
codebook. The reason for changing bases was legitimate output-range coverage;
the change also replaced an already conditional summarizer with a model whose
new conditioning behavior had to be learned.

The current [source adapter implementation](../../experiments/structure_net/ncm_lm1b_source_attention.py)
reads frozen 3,072-wide Llama features only after generator blocks 8, 10 and 11
(zero-based). The prefix remains unchanged. This gives the source three late
entry points into a pretrained unconditional generator. It is an architectural
adaptation; freezing the donor Llama does not preserve its language-generation
behavior through this new path.

The latest run trains the generator and source modules, totaling 105,323,136
parameters. It freezes the Llama body, codec and word-source readout. Thus it
would also be incorrect to explain failure as training only a tiny adapter.
The source-tree/every-layer variants were separate experiments; the latest
graded model did not incorporate them. Their uncompleted semantic comparisons
cannot be cited as established successes or failures.

The existing [frozen XSum bridge](2026-10-27_ncm-xsum-prefix-bridge.md) cut feature
MSE roughly in half against its mean control, yet preserved only 16.17% of
native content IDs. [Output distillation](2026-10-27_ncm-xsum-output-distillation.md)
raised that to 16.83%. This is direct evidence that a smaller representation
error did not preserve the downstream computation. The comparison does not
settle whether jointly learning the full conditional model would work.

**3. Reconstruction quality and generation quality are separate objectives.**

There is a particularly relevant new primary-source update. On October 28,
the [NCM author explained](https://github.com/HazyResearch/ncm/issues/16#issuecomment-6470658660)
that fixing codebook restarts improved reconstruction and utilization but
worsened downstream generation in their experiments, so they did not keep
restarts in the released recipe. This is an author report, not an ablation we
have independently reproduced.

[Ramanujan et al., When Worse is Better](https://arxiv.org/abs/2412.16326)
study the same general tension in visual tokenization: maximizing autoencoder
fidelity independently of the generative model can create latents that are too
difficult for the generator to model. Their causal regularization improves
generation by making latents easier to predict. This is evidence for the
principle, not proof that their image method transfers to our text model.

Applied here, the unresolved question is whether source features can predict a
coherent hierarchy, not just whether the hierarchy can encode a known summary.
Our decoder improvements left encoder/codebook assignments unchanged. They
therefore did not make the generator's code-label problem easier. The native
codec also retains real generalization errors, particularly entities and
numbers; it is a remaining quality concern, but fixing it alone did not rescue
the generated codes.

**4. The training scale is very different, but scaling alone is not a diagnosis.**

The [paper's Appendix C](https://arxiv.org/html/2610.14162v1#A3) lists 350,000
conditional XSum updates at batch 512 after unpaired training. That is
179.2 million sequence presentations. Our latest adaptation has 6,416 updates,
effective batch 32 and 205,248 presentations: approximately 873 times fewer
presentations for this stage. Its peak learning rate is 3e-5 versus the paper's
3e-4. These comparisons exclude earlier local training and inherited pretraining,
and involve different tasks, target lengths and initialization. They are not
total compute ratios or estimates of a required budget.

The [paper's conditioning recipe](https://arxiv.org/html/2610.14162v1#A6.SS3)
uses corrupted earlier codes with clean targets, and says model-error-based
corruption is future work. Our latest run already uses equal loss per level and
uniform replacements decreasing from 25% to 5% across levels, on half of
presentations. Neither adding basic noise nor equal-level weighting is an
untried missing ingredient.

Only four recovered-source presentations per source were scheduled. Previous
public training did include 162,353 pairs; on a fixed public TRAIN probe,
[shuffling sources did not reduce code agreement](2026-10-27_ncm-public-pair-fit.md).
A [source-blind positional-mode baseline](2026-10-27_ncm-public-code-prior.md)
also matched more reference codes than the epoch2 matched-source model.
Aggregate code agreement can reward frequent labels without establishing
conditional competence. More data is reasonable to investigate only after
useful source dependence is demonstrated on a controlled comparison.

**5. Target variability is a newly quantified secondary difficulty.**

The offline analysis groups all 43,894 recovered pairs by exact native prompt
hash, source alignment hash and source-text hash. It does not conflate the same
text at different dates or with different role conditioning. Every saved code
record is checked against its individual SHA-256 and frame/pair bindings.

| Existing TRAIN metadata/code measurement | Result |
| --- | ---: |
| Exact conditioned inputs | 40,890 |
| Inputs with one target | 37,892 |
| Inputs with two targets | 2,992 |
| Inputs with three targets | 6 |
| Inputs with multiple targets | 2,998 / 40,890 = 7.33% |
| Target-variant pairs compared | 3,010 |
| Variant pairs with different root codes | 2,915 / 3,010 = 96.84% |
| Differing code positions across all variant pairs | 87.26% |

These are alternative discrete labels, not necessarily contradictory facts or
invalid summaries. They demonstrate that the coarse code is often sensitive to
which target rendition was chosen, even with identical conditioning. The study
does not quantify broader semantic diversity or prove that this caused failure.
Since 92.67% of inputs have only one target, this cannot explain the whole result.

[Zhou, Neubig and Gu](https://arxiv.org/abs/1911.02727) show that parallel
translation models benefit from sequence distillation that reduces target
complexity, and that the suitable complexity depends on student capacity.
[Kim and Rush](https://arxiv.org/abs/1606.07947) distinguish sequence-level
distillation from matching token probabilities. The applicable hypothesis is to
train a bounded comparison with one consistent, verified summary per input,
not to assume that more historical variants always provide useful supervision.
It remains a hypothesis for NCM and should be tested against the same data and
budget with variant rotation retained as the control.

The old 42.19% joint label-review result covered a broader mixed cache. It is
not a measured quality rate for the subsequently filtered 43,894-pair corpus.
Substring exclusions and restored metadata removed known problems, but the
retained historical labels still have no established whole-corpus semantic
quality rate. Source/target provenance verification cannot establish label
adequacy by itself.

**6. Several plausible fixes have already failed and must remain negative evidence.**

| Proposed explanation or remedy | Existing evidence | Consequence |
| --- | --- | --- |
| Add training on generated histories | Both matched arms remained 0/128 | Do not present this as an untried fix |
| Recompute residual targets after generated errors | Same 0/128 endpoint; lower TRAIN word agreement than fixed targets | Correction capacity did not become conditional competence |
| Correct only the first few codes | Supplying the first 25 correct codes raised word agreement only from 5.85% to 8.15% on the earlier reserved probe | Root selection alone does not explain all damage |
| Add published frequency/adjacency penalties | Three earlier policies all scored 0/64 | Repetition penalties did not repair missing facts |
| Increase decoder reconstruction | Four current controls reconstruct exactly with correct codes but fail with generated ones | Decoder-only work cannot solve this predictor failure |
| Demonstrate tiny-set memorization | An older variant reached 99.67% free code agreement on 32 TRAIN examples | Generic tiny overfit is already established for that variant |

The corresponding records are [generated-history training](2026-10-25_ncm-onpolicy-residual.md),
[oracle prefix](2026-10-27_ncm-oracle-prefix.md),
[decoding policies](2026-10-22_ncm-decoding-and-residual-capacity.md), and
[training competence](2026-10-22_ncm-training-competence.md). Different model
versions and populations prevent combining their percentages into one trend.

[Infinity](https://arxiv.org/html/2412.04431v2#S3.SS4) provides a genuine
related correction mechanism: alter earlier bit codes and re-quantize later
residual targets. Its image ablations distinguish this from noise alone. We
already tested a local generated-history residual-retargeting adaptation; its
negative result rules out describing Infinity's idea as a newly discovered
automatic cure. A closer implementation or larger-budget test is still an
untested proposition, not grounds to disregard that failure.

[GLAT](https://aclanthology.org/2021.acl-long.155/) provides an alternative
curriculum for learning dependencies in parallel generation. Its success is in
translation, and injecting target information during training is not equivalent
to source-only inference. A NCM adaptation would need a controlled test showing
that assistance can be removed without the current collapse.

**7. Recent upstream bug reports were checked for applicability.**

The official default branch remains commit
`c70274273f489abc73832b2b98584a8483b6b203` in the live repository check. Its tree
contains inference/model/evaluation code but no author training loop. Open
[training-code requests](https://github.com/HazyResearch/ncm/issues/10) remain;
we are reproducing a described recipe with custom trainers.

| Report or code concern | Actual relevance to the latest run |
| --- | --- |
| [EMA restart normalization, issue 16](https://github.com/HazyResearch/ncm/issues/16) | A real reported tokenizer-training bug. Current encoder/codebooks are frozen and restarts do not run. It does not explain the latest predictor failure. |
| [Aliased partial reconstructions, issue 11](https://github.com/HazyResearch/ncm/issues/11) | `idx_to_zhats` can return aliases. Current training uses `idx_to_ncm_input`; relevant decoding uses the final reconstruction or separate level tensors. Existing packed/explicit and future-code checks provide additional evidence against this path being the present cause. |
| Squared-distance tied-head logit scale, discussed in issue 16 | Current released LM1B config selects a shared linear output head. The tied-codebook branch is not used. |
| Learned root constructed under `no_grad` in an old wrapper | The active loader installs `ncm_trainable_example_batching.teacher_logits`, keeping the root and concatenation outside `no_grad`. The optimizer checks nonzero root gradients. The old wrapper alone would give a misleading code audit. |
| Frozen generator accidentally left in evaluation mode | Active `ncm_sequential_levels.configure` explicitly enables generator training and freezes only protected paths. |

Seven inspected local mechanism/training sources match their frozen hashes in
the latest training producer. This is a targeted static audit and reuse of
existing numerical checks, not a proof that the entire implementation is bug-free.
No new applicable implementation defect was established by this review.

**8. The evidence supports the method's feasibility, but not our quality target.**

Our untouched release already matched the author's inference outputs and a
128-case original-domain check reached ROUGE-L 27.8859. An external practitioner
[reports full-test XSum ROUGE-L 27.93](https://github.com/HazyResearch/ncm/issues/9),
close to the published value. That issue also contains the author's clarification
about precision, caching and timing. These are inference-reproduction evidence;
they do not reproduce training or establish accuracy against our rubric.

An independent [MLX implementation](https://github.com/janmus1c/ncm-mlx)
now provides an executable trainer and reports successful synthetic-language and
Sudoku learning. Its reported best Sudoku rates are 98.10%, 83.40% and 40.30%
for easy, medium and hard boards. It explicitly does not establish real-text
summarization and has different architecture, scale and data. Its
[training loop](https://github.com/janmus1c/ncm-mlx/blob/main/ncm_mlx/training/generator.py)
is useful as an additional implementation reference, not a drop-in successful
summarizer. The README, trainer and configuration inspected here are saved with
their GitHub blob IDs in the research artifacts.

Crucially, our separate [native XSum semantic probe](2026-10-27_ncm-xsum-author-semantics.md)
passed only **3/128** with the author's text renderer. Readability passed on
33/128. These public TRAIN sources were excluded from bridge fitting but may
have appeared in release training. They are a different population from our
application DEV and do not give a universal native-model accuracy. They do
demonstrate that successful ROUGE reproduction is insufficient evidence for
95% complete-summary quality. The rubric permits paraphrases, and renderer
repair did not change this conclusion. Automated judge validity is not human
ground truth; any future calibration should retain these results and add a
separately labeled blinded review.

**Decision and the most informative possible restart.**

Continue to shelve the current scaling run. There is no established single
training trick that turns its present outputs into a reliable summarizer.
Additional layers, codebook utilization, raw corpus size, lower teacher-forced
loss and higher reconstruction accuracy are inadequate restart criteria.

If this research is reopened, isolate source conditioning first. Use the same
intact released conditional checkpoint, original codec, 512/48 diagnostic range,
clean source/summary pairs, optimizer budget and target policy in two arms:
native source prefix versus the proposed Llama conditioning. Preserve a no-update
native control and allow the same generator parameters to adapt in both arms.
Measure source-only TRAIN and held-out summaries, matched versus swapped-source
effects, and per-level teacher/rollout gaps at fixed checkpoints. Do not jointly
change the codec, range, supervision corpus and architecture in this comparison.
This is a narrow diagnostic, not coverage of the application's 128-token target
or up-to-6,300-token prompt range.

If the native arm learns and the graft does not, prioritize the interface. If
both fit TRAIN but fail held-out, prioritize supervision/generalization. If both
fail to learn, audit objective and optimization against the independent trainer.
Because the unadapted native semantic probe is itself weak, a separate intact
pretrained summarizer should remain the practical task-quality reference; merely
preserving NCM's existing outputs cannot qualify the 95% goal.

My earlier suggestion to repeat a generic tiny-set fit was too broad: such
controls already passed on earlier variants. The missing evidence is a matched
conditional-transfer/generalization comparison, not another isolated component
success. No experiment described in this paragraph has been launched.

All new research artifacts are under
`data/experiments/ncm_paper_regime/20261030_ncm_failure_research_v1/`:
the retrieved paper, official repository/issue snapshots, independent trainer
snapshots, `analyze_existing_records.py`, `offline_diagnostics.json`, targeted
source bindings and actual analysis execution receipt. The offline calculation
used locked Pixi with CUDA disabled; it read existing TRAIN records only. No
model weights, completed judgments or prior experiment artifacts were changed.
