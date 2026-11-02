# Factual errors remaining after clean native summary training

Date: 2026-10-23. Posthoc paired development analysis. The isolated clean
native checkpoint scored **56/64 complete-summary passes, 87.5%**, under the
qualified Terra judge. All 64 judgments were valid. Faithfulness passed on 56
cases, salience on 63 and readability on 64; there were no empty summaries or
storage overflows. The 95% development gate failed, so independent Sonnet grading was not admitted.
This is a native quality-teacher control, not an NCM replacement result.

On identical sources and the same rubric, unchanged 3B scored 52/64 and the
converged shared-adapter hybrid 41/64. The matched transitions show that the
clean adapters do not improve every case:

| Earlier model | Both pass | Earlier fail becomes pass | Earlier pass becomes fail | Both fail |
| --- | ---: | ---: | ---: | ---: |
| Unchanged 3B | 46 | 10 | 6 | 2 |
| Converged hybrid | 40 | 16 | 1 | 7 |

The observed gain over unchanged 3B is four cases, or 6.25 percentage points,
on reused development data. It does not establish a general improvement or
95% final accuracy. All eight new failures include a faithfulness failure.

| Source distinction | Failed summary behavior |
| --- | --- |
| Burglars enter a blind man's home | Calls the burglars blind |
| Quotas decrease earnings and can increase hardship | Compresses this into decreasing earnings and hardship |
| Grooming recommendations include a $70 kit | Gives the overall range as $75–$80 |
| Produce may look imperfect despite high quality | Describes imperfect produce quality |
| Removable decor includes a floor rug and a tension rod | Generalizes everything to wall decorations and changes rod to rope |
| A player served fourteen seasons and was important during the 1980s | Places all fourteen seasons within the 1980s |
| A source asks for a UI and prompt workflow | Fills a table with invented selections and omits the prompt workflow |
| More than five decades and conditional festival advice | Removes the lower-bound qualifier and broadens the advice |

These errors concern attribute ownership, causal direction, numeric coverage,
qualification, scope and following the source's embedded request. Fluency or
teacher-forced word recovery cannot establish their correction. The source
passages and generated texts were frozen before judging; no decision was
replaced and no failure was removed. Future training must use other source
groups rather than adding these development cases to training.

The proper learned-codec experiment is independent of this outcome. It asks
whether the released NCM representation can reconstruct unfamiliar complete
summaries after broader unpaired training. Correct target codes are oracle
inputs there. Improving that component does not prove that a conditional
generator will preserve source facts; a source-only NCM task evaluation remains
necessary. The original parent weights remain frozen, and both final cohorts
remain sealed.

Evidence: `data/experiments/ncm_paper_regime/20261023_isolated_quality_claim_audit_v1/result.json`
contains paired counts and hashes of every input score and packet file.
`failed_cases.json` in that root preserves all eight complete sources,
summaries, cited claims and first judgments. The primary endpoint and full
word-history replay are finalized separately by
`finalize_ncm_isolated_quality_teacher.py`.
