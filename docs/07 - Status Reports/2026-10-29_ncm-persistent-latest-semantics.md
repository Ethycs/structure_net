# Reuse measured hashes within the latest evaluation's CPU worker

The latest recovered preflight completed successfully for the fixed step6416
checkpoint, all128 DEV cases and all5903 inherited first judgments. Its actual
receipt is `20261029_latest_recovered_DEV_preflight_v1/actual_tool_receipt.json`,
SHA256 `0bd2e63b3fbe886dcda486a1ddf742e6bf8446b080123a3e0f705231b4ba4d5d`.
No latest-model DEV output or semantic outcome was known when this execution
amendment was written. The same inference pilot continues unchanged; its
supervising queue was replaced after a brief, identity-checked scheduling hold.
The hold/resume records preserve the actual process and tool session. No model
or optimizer update was repeated and no producer source was edited.

The training result references about56GB of checkpoint artifacts, and its
producer references about92GB of inputs, before earlier ancestry is included.
Starting a new full validation for every export and grading phase repeats this
disk work. Preserve all verification, using one live CPU worker for subsequent
saved-output exports, judge pilots, full grades and the paired comparison.
An optional CPU warmup runs the exact existing admission context without model
or judge calls. It can overlap the unchanged GPU generation stages.

The worker starts with an empty hash map. New proof objects share only that
live process's actually measured hashes; each keeps its own input bindings.
Reuse the original open-descriptor hashing and device/inode/size/mtime/ctime
guards, including the recent-write guard. No digest or cache is loaded from a
previous process or receipt. Every previously unseen file is read completely.
Independent audits are forbidden through the service and run through the
unchanged direct CLI in separate processes with empty caches.

Use locked Pixi and CUDA_VISIBLE_DEVICES empty for the worker and its clients.
Accept only a fixed whitelist through an owner-private Unix socket. Record an
immutable worker invocation, PID, source hash and every unique request/reply.
There is one executing request at a time and no automatic retry. Preserve
failures; a failed or lost request never becomes a new quality attempt. The
client exits successfully only after the worker function returns and the
returned result path and SHA256 match the actual artifact and exact invocation.
The actual client tool receipt is the stage's completion evidence; the worker
is intentionally still alive. Bind this execution contract and worker identity
into each producer so this is not confused with worker-process termination.

Before launch, pass six worker tests and the three original hash-cache tests.
Before admission warmup, require an actual private-socket startup probe and
successful client tool receipt. It verifies the same worker invocation, an
empty measured-hash map, no CUDA initialization, and a hash-bound completed
reply without any model or judge call.
Retain the earlier eleven evaluation-adapter tests and original source-only,
readout, semantic and route qualifications. Freeze the new worker source, tests
and this plan before serving. The worker cannot train, generate predictions,
run an independent audit, change readouts or open Final320. Generation remains
in the original GPU process. Keep the same output roots, semantic protocol,
first judgments, invalid-attempt budgets and dual-judge gates. This is an
execution change only and establishes no model accuracy or speedup.
