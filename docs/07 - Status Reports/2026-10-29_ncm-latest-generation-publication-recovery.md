# Repair active output publication without changing predictions

The latest checkpoint's original two-case inference pilot exited1 at actual
tool chunk2463c4/session27411. The second progress write used the immutable JSON
helper's exclusive creation mode on the existing outputs.json. Preserve
`20261029_latest_recovered_DEV_inference_pilot_v1`, its producer, first saved
parent output, failure and actual receipt (SHA256
`168c0221e8564f7dc09c634ba331de71edbba38dd4d8e22cafccc48e8a065e1c`).
There was no semantic judgment, completed pilot or quality result. This is an
I/O lifecycle failure, not an accuracy failure. The qualified preflight remains
unchanged and usable.

Run a new inference pilot and full generation under their v2 roots. Retain the
exact same checkpoint, codec states, source cases, feature payloads, FP32
computations, rendering, readout order, memory limit and determinism checks.
Only writes to the current stage's outputs.json may replace an earlier progress
snapshot. Serialize strict JSON to an exclusively created pending file, flush
and fsync, then atomically replace the progress file. Every other artifact still
uses exclusive creation. Refuse replacement once result.json exists; a completed
root is never reused. Preserve a failed serialization's prior complete snapshot
and pending evidence. No automatic restart or retry.

Pass four publication/recovery contracts and the three original recovered-output
contracts before launch. Bind this repair, its tests, CPU receipt and the exact
failed-v1 evidence into the producer. The repeated two-case GPU pilot must
complete and reproduce the first previously saved parent output exactly before
full generation is admitted. The successful tool receipt additionally binds
that parity check. All later consumers must use the v2 generation roots; the
preflight and semantic roots retain their original identities.

The first persistent CPU worker was stopped during admission warmup because it
was bound to the failed v1 generation roots. Its successful neutral probe and
intentional retirement are retained; no model or application judge call ran in
that worker. A separately qualified scoped worker will start empty, consume
these recovered roots, and run both readouts' exports and semantic phases.
Independent audits still run in fresh direct-CLI processes. The readout accuracy
definitions and sealed Final320 boundary remain unchanged.
