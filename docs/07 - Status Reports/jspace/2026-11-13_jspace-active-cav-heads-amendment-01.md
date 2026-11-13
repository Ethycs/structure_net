# Active CAV/head study: prospective interpretation amendment 01

Recorded during training capture, before fitting, evaluation capture or
evaluation outcome inspection. The original protocol and producers remain
unchanged. This amendment only tightens the interpretation of country
specificity; it cannot turn a failed causal or head-quality gate into success.

The base protocol asks whether full-country CAV transfer changes the answer
when the queried country changes, while retaining the answer under a donor
that changes only the colour assignment. However, retention under the latter
edit is interpretable only if the answer-only donor is manipulable at this
position. An early block may still retrieve its answer from unchanged source
facts later, even after copying the entire question pattern.

Additional requirement: **dense answer-only donor transfer at block 24 must
achieve at least 80% target success on source/answer-donor clean-correct pairs**
before country specificity can be supported. These dense-control measurements
are already in the original schedule; no new examples, edits or tuned choices
are introduced.

If the base full-country causal gate fails, retain that classification. If it
passes but this dense answer-only control fails, classify the result as
`CAV_transfer_supported_country_specificity_unresolved_dense_answer_control`.
Otherwise retain the base classification. The block-29 measurements remain
explanatory. Head fitting, candidate selection, evaluation and optional
benchmark criteria do not change.

The amendment has a separate source/protocol hash and a durable freeze that
refuses to run once the evaluation directory exists. It produces an append-only
amended result and aggregate after the original aggregate completes. Reports
and registry entries must identify both fingerprints, the base classification
and the stricter interpretation; no original artifact is rewritten.
