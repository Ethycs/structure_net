# Preserve recovered-label judgments in subsequent model evaluation

The fixed recovered-label review is an intervening branch after the completed
epoch2 original and author-rendered probe judgments. Its model-independent label
scores must never enter application model accuracy, but its canonical first
decisions must be retained by later public epoch4/epoch8 and paired task DEV.

Add a new consumer wrapper without editing any running training producer or
historical evaluation module. Install the existing author lineage first, then
extend its epoch2 result with the completed, independently replayed recovered
label review. Require exact equality to that review's inherited cache, preserve
all inherited entries, and permit all and only the fixed128 packets under the
two current qualified judges. Invalid exhausted firsts remain preserved. Require
actual successful terminal receipts for prepare, pilot, grade and audit before
using the branch. Do not alter historical epoch1/epoch2 evaluation identities.

A CPU preflight must verify the complete resulting cache and unchanged128 DEV
source membership before later model judging. The wrapper must also be installed
for raw semantic audits, so report and audit consumers see identical ancestry.
Its own append-only root is `20261027_recovered_review_lineage_preflight_v1`.
Add focused tests of insertion order, idempotence, absence on earlier milestones,
and rejection of missing/changed/extra firsts. No model calls occur in preflight.

For public milestone aggregate reports, preserve the original probe, author
rendering and recovered-label additions as separate extension records. A report
consumer that assumes only the first two additions cannot report the new branch.
Report integration must be completed before publishing an epoch4/epoch8 aggregate.
Neither sample label grades nor codec reconstruction counts establish model
accuracy. Training and source-only inference endpoints remain unchanged.
