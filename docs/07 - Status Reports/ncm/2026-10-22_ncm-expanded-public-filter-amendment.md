# Expanded-data exact-public eligibility amendment

Date:2026-10-22, before any new teacher request or new data output.
The first expanded-data preflight stopped at the canonical public-source guard:
one selected saved turn was absent from the SHA-verified public turn index.
No source data was sent to a model. Preserve the failed source/plan versions in
`20261022_expanded_preflight_v1`; do not treat missing proof as a quality failure
or infer that the record is private or invalid in its original application.

Refine source eligibility before selection: every supplied source span must
match the published dataset's session ID, role, full-turn SHA, character range,
span SHA and reconstructed source text. Build the exact lookup once. Reject
unmatched rows before reserving their groups, then take the first8,212eligible
rows in the same fixed hash order. Record only IDs/hashes/reasons for exclusions,
never send unmatched source text. Run the original canonical verification again
on the complete chosen set before creating a data producer or making requests.

All counts, group/session/hash exclusions, length limits, pilot20versus train8192,
teacher configuration and future audit remain unchanged. Audit128uses the first
128hash-ranked eligible labels; its separate systems pilot uses ranks128–131,
so systems decisions never duplicate audit candidates. There are no model
quality outcomes from this new data study yet, and no semantic exclusions.
Use fresh `20261022_expanded_{data,teacher_pilot,teacher,audit_pilot,audit}_v2`
roots. Freeze/copy this amendment and the refined selector in every producer.
