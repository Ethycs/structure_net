# Complete summary quality after generated history word training

Date: 2026-10-23. Hypothesis `ncm-reviewed-draft-history-word-v1`.
Classification **generated_history_development_quality_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_reviewed_draft_history_results_terminal_v2/campaign_results.json`,
SHA256 `9560dd529786efa588cd9416a6112481b51694831a7c1ddab0bfc0fb5127abd4`.

| Development cohort | Judge role | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
| short_development | secondary | 0/64 | 64/64 | 0.00% |
| long_diagnostic | secondary | 0/64 | 64/64 | 0.00% |

Every summary requires supported material claims, useful salient content,
readability, nonempty text and at most128stored cl100k tokens. Sonnet is admitted
only after64valid Terra judgments and at least61passes in that cohort. All64
cases remain in each denominator. These reused development cohorts are source
disjoint from training; both final cohorts remain closed. Wilson intervals,
component failures, overflows and invalid judgments remain in the aggregate.

The original Llama body and MLPs stayed frozen. The registered16,384-step concept
forecaster generated its own eight coarse levels from source inputs only.
A separate word branch trained for exactly32,768updates on690complete
vetted TRAINsummaries, using actual generated histories plus a0.25clean-history
auxiliary. The forecaster, original codec and unused word-branch concept heads
stayed fixed. Word gradients did not alter the forecaster. This is an NCM
adaptation with two new-head parameter sets, not a paper training replication.

All sampled sources, full frames, code-history hashes, learning rates, optimizer
steps and sampler state replayed. Actual CUDA pilots, model/optimizer/RNG and
composed inference reloads passed before downstream stages. Only the fixed final
word checkpoint was selected. The two first inference outputs were retained
exactly. Every192-position raw word frame and predicted concept level remains
available; texts use one decode at the first global EOS. No oracle concepts,
gold history, output repair, lookup, autoregressive fallback or best-of selection
entered evaluation. All prior first judgments and new negative decisions remain
exact and were replayed from raw responses. Individual failures are retained.

This development result does not establish95%final task accuracy. All earlier
failed models remain failures. No end-to-end throughput, latency or compression
benefit has been measured. The overall goal remains unmet pending qualifying
quality on the sealed source-disjoint final evaluation.
