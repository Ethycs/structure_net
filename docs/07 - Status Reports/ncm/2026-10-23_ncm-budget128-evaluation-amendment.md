# Evaluate the application's128-token summaries without a32-word target

Date: 2026-10-23. EXPLORATORY evaluation amendment
`ncm-budget128-judge-profile-v1`. Existing completed experiments remain unchanged.
During inspection of the10-candidate native-label diagnostic, one salient=false
judgment explicitly penalized a supported summary for exceeding approximately
32 words. The inherited rubric's salience paragraph contains that budget, while
the user's application stores up to128 tokens. The32-word extractor atom target
is a separate operation. The old rubric is therefore mismatched for summaries
that legitimately use more of the128-token storage budget. A second diagnostic
failure is a real recipe-order error and is not excused by this amendment.

Make one explicit rubric change: evaluate main point/useful essential details
within the128-token storage budget; impose no32-word target or length penalty
inside that cap. Preserve faithfulness, salience coverage, readability, supported/
contradicted/insufficient-evidence claims, exact source-line evidence, nonempty
text and the existing code-enforced128-cl100k-token ceiling. No model outputs,
generation settings or first old-profile judgments change. Old0/64 language
failure remains a failure and is not explained by this budget discrepancy.

Before new-profile quality claims, requalify both existing judge aliases on all
48 fixed source-reviewed calibration candidates from12 nonevaluation source
sessions plus two new long-budget controls on calibration source0: one literal-
fact-supported75-word coffee summary and the same summary with an invented
terminal. These are authored/source-reviewed controls, not independent human
gold. Require all50 decisions correct and all requests valid per alias; retain
each first valid result, maximum three transport/schema attempts, four workers,
no semantic retries or judge substitution. Use a separate root per judge.

New roots `20261023_budget128_judge_{secondary,primary}_v1`. Hash this amendment,
the profile implementation, original calibration inputs and exact new packets.
If either qualification fails, preserve it and stop new-profile task claims;
do not change expected labels to match the judge. Report component decisions and
all50 outcomes. A new-profile re-evaluation must use fixed existing outputs,
keep all prior-profile scores side by side, and never replace their cache.
Qualification alone says nothing about NCM model quality or the95% objective.
