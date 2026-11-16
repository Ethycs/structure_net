# Broad-response progressive rebuild: launch status

`JSPACE-QWEN3B-PROGRESSIVE-RESPONSE-REBUILD-01` is running as an exploratory
one-block, then conditionally two-block pilot. No recovery or quality pass has
been established. The user required preserving the all-at-once run.

The new branch starts from original Qwen2.5-3B-Instruct and uses the admitted
eight-family complete-response corpus for fresh functional geometry and
distillation. It fits layer35 first, then may prepend34 after a fully eligible
packed evaluation. There is no automatic extension to36 blocks. The
[protocol](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-progressive-response-rebuild.md)
and [systems amendment](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-progressive-response-rebuild-systems-amendment.md)
fix the sequence and prospective native-relative gates.

The data plan contains128 balanced TRAIN calibration cases and4,350 response
positions including EOS,32 disjoint TRAIN diagnostics,256 group-disjoint local
alignment cases, and736 joint-recovery examples totaling25,172 response tokens.
Joint checkpoints occur at updates10/19/46, after5,490/10,440/25,172 tokens.
Means use all TRAIN native inputs; relation fitting uses the compiled interface.
Neither exposed MC DEV nor response DEV is used for fitting. Stage recovery
requires a discarded actual CUDA pilot and fresh scientific initialization.

Twenty-nine focused tests passed before launch. The first actual capture found
a construction-context bug before capturing any example: inference tensors did
not expose version counters. The correction retains ordinary versioned tensors
and the unchanged immutability checks. The expanded data suite passed11 tests,
including the real capture lifecycle and rejection of a buffer mutation that
left outputs unchanged. Together with the unchanged20 other checks,31 focused
checks cover this implementation. Producer commits are `6296e22`, `abd13ad`,
and corrective `27bf58e`.

Failed v1 artifacts remain under
`data/experiments/jspace_3b_progressive_response_rebuild/20261116_v1`.
Its first controller request timed out at Infisical before acquiring a lease;
its retry exposed the construction bug. Neither attempt trained weights or
produced usable native banks. No failed attempt is counted as quality evidence.

The corrected v2 data root is
`data/experiments/jspace_3b_progressive_response_rebuild/20261116_v2/data`,
fingerprint
`114d644102a18f97228ea8867db95cf5a8e0e321078835a7cdf5dd8781de8dd5`.
Its controller launched2026-11-16T09:16:11Z after confirming no duplicate worker
existed. Native capture received3060 lease `2baac73825a9`,11264MiB allocation
and11008MiB allocator cap. It passed the earlier failure point and had completed
64/4,569 response sequences with exact native teacher/boundary checks at this
entry. Banks34/35/36 plus the separate MC boundaries require about8.4GiB.
The controller will run fitting, a discarded CUDA pilot, then bounded recovery
under separate leases, stopping on execution or retention failure.

The superseded progressive16 worker was interrupted cleanly after its update90
checkpoint and grading were durable. Update90 scored171/256(66.80%) on exposed
MC DEV, versus186/256(72.66%) initially; response NLL improved6.96739to3.97192.
It did not qualify. The worker reached update95 before the final judgments
returned; those additional unsaved updates are not retained checkpoints. The
original initializer and saved checkpoints0/46/90 remain available. Its old
3060 lease was released; v2 reuses the completed transition receipt read-only.

All-at-once recovery remains unchanged on2060SUPER, worker558959 and lease
`0d378d7d602e`. At update46 it scored91/256(35.55%) versus109/256(42.58%)
initially, while response NLL improved13.46914to7.59897. Generation/grading of
that checkpoint was still running at this entry. Lower loss does not rescue
lost benchmark accuracy. The separate packed-checkpoint observer also remains
active on the plain2060; no matched deployment speedup is claimed here.
