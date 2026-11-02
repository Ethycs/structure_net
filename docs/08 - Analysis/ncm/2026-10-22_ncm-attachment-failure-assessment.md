# The current NCM attachment fails the summarization task

Date: 2026-10-22. This is an assessment of the existing negative evidence, not a
new accuracy result. The current frozen-Llama/NCM attachment is a complete
practical failure on the measured development task. Paper-replication
terminology does not reduce the severity of that result.

The final checkpoint produces0/64 passing unseen summaries under the qualified
Terra rubric. All judgments were valid, with no retries; all64 outputs fail
faithfulness, salience and readability. The same-source original Llama scores
50/64. Primary Sonnet grading remains pending, and the final test remains sealed.
These are model-judge results, not independent human annotations.

Direct inspection makes the failure apparent without a subtle quality rubric.
The first development source gives Badlands travel advice, while its generated
summary begins `Tipslands's Badlandslands pathlands`. The second source offers
books for a solo hike; its summary includes `confams` and `legacy sitesork`.
These are the preserved final outputs, not newly selected generations or
retries. The model often produces broken language as well as unsupported facts.

The98.1237% target-token reconstruction on128 training cases establishes that
the optimization can fit those examples. Shuffling either the Llama features
or the native text substantially damages that reconstruction, showing that
the fitted model uses both channels. Neither measurement establishes usable
summarization on a new passage. The strong training/development gap supports
memorization and unsuccessful task transfer. It does not establish whether the
dominant fault is the source interface, learned source-to-code mapping,
off-manifold predicted codes, decoder transfer, data coverage or an interaction.

The published NCM has its own trained source encoder, generator and text codec.
Our experiment changes the source interface, domain, supervision and training
recipe. Attaching this head to a frozen Llama does not by itself preserve the
parent's output behavior. A successfully fitted code predictor on known targets
is insufficient evidence that this new attachment transfers the parent's
language ability. Describing intermediate reconstruction as model accuracy was
misleading; actual unseen readable and grounded output should have been an
early prerequisite.

Hold further scaled student training until transfer controls qualify it.
Existing fixed mechanism diagnostics and public-label preparation may finish;
they cannot establish95% task accuracy. The next attachment diagnosis should
first test reconstruction of *unseen* known target texts, then compare free
generation with target-aware coarse-code controls on the same unseen sources.
That distinguishes a failing codec from source-to-code prediction errors. Any
new source-interface training comparison must isolate that intervention and
show real development-summary quality before expansion. No remedy,95%quality,
smaller complete model or speedup is established.

The32-case author/local inference agreement remains a useful implementation
control. It is not a full published-score or training replication. A new
batch32 benchmark pilot preserved32 nonempty outputs but stopped at its declared
cross-batch exact-output gate:31/32 texts match the earlier batch4 run; one
changes punctuation and inserts `said`. The original failure, source snapshots,
command, log and author output remain at
`data/experiments/ncm_paper_regime/20261022_released_xsum_benchmark_pilot_v1`.
The merge/scoring stage and full11,334-case benchmark did not execute. This
pilot is an execution/numerical-parity failure, not evidence about the paper's
full ROUGE scores or a remedy for the application failure. No rerun or amended
success is included here.

Evidence: [measured generalization diagnosis](2026-10-22_ncm-generalization-diagnosis.md),
`20261022_full_train_v1/development_step5120.json`,
`20261022_generalization_diagnosis_v1/campaign_results.json` under the paper
artifact root. [Published architecture and metrics](https://arxiv.org/html/2610.14162v1),
[official evaluator](https://github.com/HazyResearch/ncm).
