# Curation provider recovery after fresh-case timeouts

Date: 2026-10-22. Exploratory amendment to `ncm-paper-regime-recovery-v2`.
The32-worker pilot passed, but fresh-case screening repeatedly hit180-second
read timeouts. Stop only that worker and preserve its results, including every
valid negative. The prior8-worker run had329written decisions; the later32-worker
run added a few valid replies before termination. Neither full screening pool
completed. The v2stage driver correctly stopped without conditional training.
This is an infrastructure failure, not evidence of bad model accuracy.

New root `20261022_curated_v4` uses the responsive qualified primary judge at
eight workers for previously unjudged labels. The primary judge authored the
labels, so screening is no longer independent of their author. This limitation
is explicit. Require a32-label post-selection audit by the qualified secondary
judge with>=31passes before conditional training. It is independent of the
author and most new selections; some imported decisions originated with that
same secondary judge. Final model evaluation still uses both judges on sealed,
source-disjoint test inputs. No quality threshold is weakened.

Import the first valid original-audit decision, otherwise the first valid
8-worker reply, otherwise the first valid32-worker reply. Preserve all valid
failures. For a case with earlier completed transport/schema failures and no
valid reply, conservatively exclude it rather than resetting its retry budget.
Only never-judged cases get new primary requests (maximum3schema/transport
attempts); all1,999cases remain accounted for. Keep all accepted labels in
original order; minimum1,024. No pilot decisions are reused for selection.

The new8-worker pilot uses32audited public sources with case IDs prefixed by
`systems-primary-v4-`, ensuring distinct request payloads from previous audit
requests. These repeats validate transport/schema only. If it fails, do not
start the full new-provider screening stage. Preserve the failed32-worker
experiment and the stopped v2driver. No GPU-service changes or speed claim.
