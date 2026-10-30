# Guarded hash reuse in the legacy grading verifier

Both128 paired outputs completed at actual tool chunk61fb06/session69850. The
v2 CPU worker completed the standalone-codec pilot/full exports at chunks741764
and977279. Its judge pilot is running; no completed latest-model semantic score
has been observed. Both independent auditors have completed their own admission
and are waiting for actual completed grades. The inspected model examples remain
poor; this change does not respond to or revise any judge decision.

Read-only inspection identified a remaining uncached path:
`ncm_attention_repair_data.completed` calls `ncm_paper_data.verify`, whose imported
`file_hash` rereads all bound training checkpoints. The live CPU worker's cached
SHA helper does not affect that imported alias. The current judge-pilot process
is preserved and allowed to finish without modification.

For future stages, use a fresh v3 CPU worker. During each request, temporarily
bind only `ncm_paper_data.file_hash` to the same worker's original stat-guarded
measurement method. Restore the alias on return or exception. Keep the verifier,
expected digests, error decisions, source bytes, rubric, first judgments and
candidate order unchanged. First access still reads the complete file; cache
hits require the original identity/time guards. Never seed the new worker from
the v2 worker, auditors, disk or declared receipts. The v3 worker starts empty,
passes its neutral actual-client probe, and warms its own admission while the
unmodified v2 judge pilot finishes. Close v2 cleanly after its actual terminal
pilot receipt; preserve every pilot first and continue grading in v3.

Pass four focused legacy-verifier contracts and the three existing hash-cache
contracts before launch. Also require the unchanged scoped-worker and core-worker
qualifications. Bind this adapter, tests, CPU receipt, plan and distinct worker
invocation into future producers. Audits remain in their existing separate
processes, using their own measurements; they never use this semantic worker's
cache. No model is regenerated or retrained, and no semantic packet is rerolled.
This changes verification cost only, with no inference-speed or quality claim.
