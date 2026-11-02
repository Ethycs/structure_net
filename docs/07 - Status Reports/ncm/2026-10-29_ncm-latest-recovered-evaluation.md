# Evaluate the latest recovered checkpoint next

The user explicitly requested grading the latest 43,894-pair model after the
older 2,384-pair candidate finished at 0/128 complete-summary semantic passes.
The older candidate's independent audit confirms all 128 valid failures and
5,903 retained canonical first judgments. Both repaired judge routes passed
50/50 controls and independent raw-response audits. The latest model's four
inspected TRAIN rollouts remain poor; its DEV outcomes have not been generated
or inspected. The improved standalone codec reconstructs those four references,
which is a component diagnostic rather than task accuracy.

This dated amendment prioritizes the fixed recovered checkpoint before the
unstarted tree and dense candidate evaluations. It supersedes only that ordering
in the October 28 candidate-route plan. Require every deferred and superseded
evaluation root to remain absent; reject any already started branch rather than
discarding its first judgments. Preserve the completed existing and expanded
task arms and all earlier judgment history. Any later evaluation of the deferred
candidates must extend the new history explicitly.

Use the unchanged recovered training/evaluation design: audited final step6416,
checkpoint SHA256
`edc88690abaa22b29989be5042269d86f011483a76c8f7b473e4e21b41e30799`,
all 64 short and 64 long source-disjoint DEV cases, original prompts and cached
frozen-body features, FP32 source-only prediction on physical GPU0 RTX3060,
unchanged rendering and a 128-token stored limit. Compare the same two readouts
in the fixed order: improved standalone step6212 decoder, then the original
decoder with its source readout. Generate one shared set of source-predicted
codes; neither generation path receives reference words or codes. No weights,
training, source clipping, threshold, judge alias or rubric changes are allowed.

Reuse the existing generation, export, semantic scoring and independent audit
implementations through a process-local adapter. New artifact roots begin
`20261029_latest_recovered_`. Bind this amendment, adapter, focused tests and CPU
receipt into every new producer. Reuse the already qualified process-local hash
policy: first access reads the entire artifact, guarded reuse only within that
process, and an empty cache in each independently launched audit.

Before preflight, pass four new ordering/provenance contracts plus the existing
three recovered-output and four semantic contracts. Then run CPU preflight,
repeated two-case inference and full128 generation. Export each fixed readout,
run its four-case Terra pilot, grade all128, and independently replay the saved
first replies before the next readout. Use the repaired qualified route, retain
all failures and the existing invalid-attempt budget. Sol remains conditional
on at least61/64 valid Terra passes in each cohort; DEV qualification still
requires the original same-case dual-judge gate. Report complete-summary passes,
denominators, validity and component scores for both readouts, followed by their
paired comparison. An invalid response is unresolved, not a quality failure.

This remains exploratory model development. The changed evaluation order cannot
retroactively confirm an older study. Final320 stays sealed and the global95%
goal is not achieved by a DEV score alone. No speedup is measured in this run.
