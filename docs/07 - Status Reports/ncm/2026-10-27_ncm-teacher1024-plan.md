# A bounded fresh batch of task-specific summary supervision

Hypothesis/workflow ID: `ncm-fresh-task-teacher1024-v1`.
This is training-data preparation, not a claim of improved student accuracy.
The latest application DEV remains 0/128. The reserved-code diagnostic found
weak unseen code prediction; the early-code correction study did not rescue
the text. The existing 1,376 reviewed training summaries span 1,376 components.
An earlier collection stopped during a service outage with all requests drained.
It left 8,487 of the original 10,000 reserved TRAIN sources unattempted.

Select exactly 1,024 of those untouched sources, using 512 from source-length
bin 0 and 128 from each of bins 1–4. Within a bin sort by SHA256 of
`ncm-fresh-task-teacher1024-v1:case:<id>` then ID. Selection uses source metadata
only. Require distinct source components and exclude every one of the 1,513
previously observed sources, including all 73 invalid first pipelines and the
64 reserved component cases. No previous failure receives another attempt.

Before requests, renew the 512-component metadata guard and the full-text
guard against the 128 open DEV and 64 component sources. Verify original
public provenance, complete 192–2,048-token inputs, and the existing source
metadata. Replay all 1,024 cached feature payloads and hashes as finite original
FP32 tensors with a 256 MiB bounded cache. No feature is recomputed or clipped.
No final-test source payload is opened or hashed. A future final-test admission
must extend its full-text guard to this newly admitted training set.

Use the already jointly qualified teacher recipe: Terra draft
`codex_sdk/gpt-5.6-terra`, Sol review `codex_sdk/gpt-5.6-sol`, unchanged prompts
and parser, source-line support links, 6,300 prompt tokens, 768 draft output
tokens, 512 review output tokens, and the same three-attempt transport/schema
budget. Keep the first valid response, including poor content. No semantic or
capacity rerolls, target editing, clipping, or fallback teacher are allowed.
Keep every raw request and response. Individual labels are not presumed
semantically qualified merely because this recipe passed its earlier study.

Run focused CPU tests, source/feature preparation, then an eight-worker systems
pilot containing the first selected source, longest source, at least one source
from every bin, and additional selection-order cases up to eight. Independently
replay its raw responses. All eight must have valid draft/review stages and
complete targets fitting both 128 stored tokens and the native 128-position
frame before full generation is admitted. The full batch reuses those exact
eight firsts and requests only the other 1,016 sources.

Use the existing bounded eight-worker queue: no more than eight pending jobs;
stop new submissions on transport/model/termination exhaustion, local failure,
or three consecutive schema-exhausted pipelines. Drain in-flight jobs and retain
all partial firsts. No automatic service restart. An isolated schema-invalid
first is retained and excluded from later training; it is not retried anew.

After collection, independently reconstruct every first candidate from its
ordered raw attempts, exact request body/model/termination, support parser,
token counts, complete native word frame, and source binding. Verify unchanged
pilot firsts and all prior history. This produces candidates only; separate
audited catalog admission and a prospective training comparison are required.
Do not change the live public curriculum or pool this preparation with its
accuracy. The 95% complete-summary semantic goal and sealed final320 stay fixed.

Fresh roots: `data/experiments/ncm_paper_regime/20261027_teacher1024_<stage>_v1`
for `prepare`, `pilot`, `pilot_audit`, `generate`, and `audit`. Code entry point:
`experiments.structure_net.ncm_teacher1024`. Durable pilot/generation launcher:
`experiments.neural_architecture_lab.ncm_teacher1024_service`. Run at repository
root through `pixi run --locked -e ncm`, CPU-only with two OMP/MKL threads.
Record actual process receipts before dependent stages; launch success alone
does not establish worker completion. Frozen source and artifacts are append-only.
