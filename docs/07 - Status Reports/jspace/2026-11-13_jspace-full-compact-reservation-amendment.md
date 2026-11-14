# Reservation readiness amendment

Study: `JSPACE-FULL-COMPACT-01`. Recorded before any full-model quality outcome.

The first fit attempt, `20261113_fit_v1`, stopped at its first layer-completion
reservation check. Its lease was active, but the controller temporarily marked
memory as not ready. The controller logged 1,431 MiB of unaccounted memory during
the release of the teacher weights. The process exited and the wrapper released
the lease. No fit checkpoint or accuracy result was produced.

Preserve that root and its frozen producer. The identical fit runs in
`20261113_fit_v2`. A new stage wrapper retries only the specific active-but-memory-
not-ready condition, every three seconds for at most 45 seconds. It performs no
GPU operations while waiting. A successful fresh reservation check is required
to continue; expired, invalid or unauthorized leases still fail immediately.
The same wrapper protects subsequent conversion and adaptation stages.

The model, task splits, Jacobian estimator, ranks, polynomial degree, channel
budget, precision, losses, optimizer and evaluation plan are unchanged. Ten CPU
tests pass, including the bounded wait, rejection of expired leases, and refusal
to continue while readiness remains false. This is a systems correction, not an
accuracy-driven change or an independent experimental replicate.

The first fit's diagnostic recorded 86.50% of block-zero Jacobian energy inside
the 1,024-dimensional common core and 965 cross-task relation reuses. Those are
partial calibration diagnostics, not a completed model result. The source
recipe's classifier accuracy remains external supplied evidence.

Controller timestamps are local America/Los_Angeles time. The first failure
record's mistaken UTC label is corrected by its separate append-only
`failure_time_correction.json`.
