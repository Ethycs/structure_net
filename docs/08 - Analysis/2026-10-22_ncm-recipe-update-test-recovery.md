# NCM recipe update: recovered candidate-level test judgments

**Updated NCM: 0/64 passing complete summaries.**
The 95% point target is not met. Date: 2026-10-22. EXPLORATORY, one seed.
ID: `ncm-recipe-update-test-candidate-recovery-v1`. Classification: `candidate_recovery_updated_below_95`.
Aggregate: `data/experiments/ncm_recipe_update/20261022_test_recovery_v1/campaign_results.json`.
SHA256: `03af3f7b3e256c27c6e61ac717bbdae8e93cf17031ae10e4b939c3b139567eb2`.
[Recovery protocol](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-candidate-recovery.md).

| Model | Passing / valid candidate judgments | Missing |
| --- | ---: | ---: |
| legacy | 0/64 | 0 |
| updated | 0/64 | 0 |
| parent | 61/64 | 0 |

The original [whole-packet comparison](2026-10-22_ncm-recipe-update-test.md)
remains invalid: one parent evaluation contradicted its own claim labels in all
three replies, although both NCM failures in that packet were valid. This new
diagnostic parses each candidate separately with the unchanged parser, original
source lines and candidate IDs/text. It uses the first valid evaluation in
chronological order, never the most favorable. No NCM output, judgment or
Boolean decision was regenerated or changed. Only the parent candidate with no
valid existing decision was eligible for a separate schema-repair packet.
Every raw reply and lineage is retained. Validation granularity changed after
outcomes were known; this cannot rescue the original invalid comparison or
the failed training-reference gate.

The 64 source-disjoint test cases are the previously inspected original cohort,
disjoint from the 32 cleaned training pairs. Judge:
`claude_code/claude-sonnet-4-6`, unchanged source-grounded faithfulness, salience,
readability, nonempty and <=128-token rubric. Public LongMemEval file SHA and
source spans were verified before repair. No private passages or original
private targets were sent. Judge calibration previously accepted24/24 good and
rejected24/24 bad agent-authored controls; this is not human-gold accuracy.
Wilson intervals in the aggregate exclude judge error and model-seed variation.

The [training update](2026-10-22_ncm-recipe-update.md) implements real nested
decoder cutoffs, same-level conditioning corruption, equal level loss and paired
codec hashes with a frozen parent body and frozen encoder/codebooks. It fitted
the small training set, but source-disjoint summary competence is not established.
Neither model has a useful test pass rate. No speedup or full application-range
claim follows from this experiment. Original XSum sanity control: ROUGE-L27.89,
codec reconstruction99.82% on128 cases; it is not a full paper replication.

The training comparison remains `recipe_update_training_positive_control_failed`:
updated29/32, legacy30/32, teacher references30/32 under a declared31/32 gate.
The separately recorded fresh-body inference check matched the cached output
on one training example; it proves integration, not generalization.

Reproduce candidate recovery from the saved raw replies with the moba Python,
memory dependencies and tiktoken cache:

```text
python -m experiments.structure_net.recover_ncm_recipe_test recover --report NEW_REPORT
python -m experiments.structure_net.recover_ncm_recipe_test register --report NEW_REPORT
```

Artifact roots are pinned in source; reproduction needs a new output root.

## Verification

Thirty-two focused CPU tests passed, including a regression that preserves an
earlier valid NCM failure when another candidate invalidates its packet and a
later reply is more favorable. The unchanged full packet remains invalid.
The CUDA training pilot, fixed matched updates, paired-codec reload, fresh-body
inference and all64-case generation paths completed. These systems checks do
not establish useful test quality or a throughput improvement.
