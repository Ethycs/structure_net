# Isolate persistent CPU request adapters

Code review before any latest-model semantic judgment identified a lifecycle
constraint in the initial persistent worker: semantic consumers install wrappers
that accept only judge-pilot, grading and audit producer stages. Leaving those
wrappers installed would reject a later saved-output export or paired comparison.
This is a prospective execution correction, not a model-quality failure. The
initial worker source remains frozen and its successful neutral probe is kept.

The original inference pilot then encountered an exclusive progress-file write
error. Preserve that failure under the separate generation-publication recovery
plan. The first CPU worker was intentionally retired during warmup, before any
model or judge call, because it was bound to those failed generation roots.
Use the new scoped worker for both readouts and their paired comparison, consuming
the corrected v2 generation roots and the unchanged completed preflight. It can
warm its own empty hash cache while GPU generation proceeds, but must not export
either readout before complete generation, nor export the second readout before
the first readout's independent audit completes.

The scoped adapter restores the producer writer, semantic configure function,
per-request proof object, artifact-hash bound method and judge request callback
after each request, including failures. Retain the original measured hash map
only inside that live process; never copy it from another worker or disk. Keep
independent audits in fresh direct-CLI processes. Request identities, all source
and readout computations, canonical firsts, result schemas and quality gates
remain unchanged. Failed stages receive no automatic retry.

Pass the new success/failure isolation contract and all nine original worker/
hash-cache contracts before launch. Bind this adapter, plan, tests and CPU receipt
into each producer. Use a fresh private socket and immutable worker invocation
under `20261029_latest_semantic_worker_v2`; require the same actual startup-probe
client receipt before warmup or exports. The worker's client also checks the
scoped adapter's source digest. This amendment claims no model accuracy or
inference-speed improvement. Both readouts retain their original evaluation
roots and order; no latest-model DEV quality result has been observed yet.
