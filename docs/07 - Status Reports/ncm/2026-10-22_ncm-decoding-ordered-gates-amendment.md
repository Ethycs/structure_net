# Ordered decoding-policy quality gates

Date:2026-10-22, after generation/sealing and inspection of the first3cases,
before any new policy's LLM quality judgments. Those inspected outputs remain
scrambled under all3policies; this is qualitative evidence, not a scored result.
Preserve the original decoding plan and all policies, including its raw control.

Evaluate all3policies on all64development cases with the qualified secondary
judge first. If none reaches61/64with complete valid judgments, the joint
development gate has already failed: omit primary grading, explicitly report it
as not run, and make no primary/joint accuracy estimate. This saves unnecessary
calls while the primary model is generating broader training labels. If any
policy reaches61/64, retain the original primary grading requirement before any
success/admission claim, after teacher completion or a combined-concurrency
pilot. Never declare success using the secondary judge alone.

The64-case cohort, rubric, pass calculation, all secondary outcomes and the
95%goal stay fixed. Missing/invalid results are not semantic failures or passes.
Preserve all valid negatives; no retries to obtain a more favorable judgment.
This amendment changes the order/conditional execution of corroborating
evaluation, not the success criterion. The final256test stays sealed.
