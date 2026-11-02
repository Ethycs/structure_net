# Curation resource amendment

Date: 2026-10-22. Study remains `ncm-paper-regime-recovery-v2`.
The original eight-worker screening job is still running. Its aggregate quality
outcome has not been computed. The codec has finished1,024updates and reached
98.2047% non-PAD reconstruction on the training labels; semantic control pending.

To reduce the labeling bottleneck, validate32 independent concurrent requests
using the same qualified secondary judge and unchanged single-candidate rubric.
Use the32already-audited cases only for transport/schema validation; repeat
decisions never enter selection. This systems-only pilot may overlap the
original8-worker job, representing an even higher connection load. No inference
speed or isolated timing claim follows. If any final request is invalid, keep
the existing8-worker job and do not launch the32-worker campaign.

If the pilot passes, stop only the identified curation worker, preserve all raw
attempts and write an interruption receipt. New root `20261022_curated_v3` imports
every first valid original audit and partial-worker decision, including valid
failures and raw valid replies that did not yet get a row file. Original audit
decisions take precedence. Never rejudge a valid failure. Cases with three prior
invalid attempts remain invalid/rejected; cases without a completed valid reply
can be requested under the new infrastructure stage. At most8in-flight remote
requests may be lost on worker termination; record that limit explicitly.

All1,999 original cases, acceptance rule, minimum1,024, public-source guard,
32-label post-selection audit, final-test split and model-quality gates remain
unchanged. This changes request concurrency and evidence reuse, not the rubric
or semantic threshold. Source and input hashes plus copies are retained in each
new root. No GPU service is modified; neural training still uses one GPU worker.
