# Preserve the teacher campaign failure and audit its valid labels separately

Date: 2026-10-22. EXPLORATORY `ncm-expanded-valid-label-recovery-v1`.
The original8,192-source teacher campaign is still live. Read-only inspection
has found one batch invalid after the declared three schema/transport attempts.
Its original requirement that every batch be valid is therefore already
unattainable. Do not restart, edit or relabel that campaign. Let its scheduled
requests finish, preserve its original failed-data-gate result, all raw attempts,
accepted labels and exclusions. No valid judgment is retried for a better score.

After the original worker is terminal, a new data-preparation study may retain
its available valid labels. Require at least7,000 eligible labels, exact source
membership in the preselected8,192, exact agreement with each batch's first
valid response, source/group uniqueness and original GPT2+EOS fit without
truncation. Independently reconstruct the accepted/excluded sets from batch
rows and verify exact agreement with the original written files. Failed batches
and labels exceeding the canvas stay excluded; no extra teacher request or
quality-based source selection. Check every original producer dependency hash.

The recovery result must explicitly name the failed parent gate and original
invalid/retried/oversized counts. Copy accepted labels exactly into a fresh root
and seal source/output hashes. This does not retroactively pass the original
campaign or establish teacher/student accuracy.

Audit the recovered labels independently with the qualified secondary Terra
judge and unchanged whole-summary rubric. Use original hash order
SHA256(`expanded-audit-v1:`+id): first128 for the audit, positions128:132 for a
separate four-request systems pilot. The pilot requires four valid judgments,
not four semantic passes. Four workers, existing schema/transport retry cap,
first valid semantic outcome wins; invalid outcomes retain bounds. Report the
audit numerator/denominator and Wilson interval. Audit point accuracy at least
122/128 is a teacher-data quality gate for a separately declared larger training
study; it is not a95%-confidence guarantee, student score or final-test result.
If it fails, preserve that result and design a new data-quality intervention.

No original-body training, feature-bank mutation, test-label opening or speed
claim. Artifacts: `20261022_expanded_label_recovery_v1` and
`20261022_expanded_recovery_audit{_pilot,}_v1` under the paper root.
Do not run preparation against a live or incomplete parent campaign.
