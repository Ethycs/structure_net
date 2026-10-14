# Optimizer stability amendment before primary training

Date: 2026-10-13. Hypothesis: `llama32-3b-neighbourhood-tensorization-v1`.
Parent: the same-date neighbourhood tensorization plan. EXPLORATORY.

Known evidence: CPU and tiny CUDA lifecycle tests passed. The full-size
rank-1024 group-0 pilot passed exact dense-teacher parity, finite nonzero factor
gradients, longest-request memory (2,315,926,016 allocated bytes), and exact
checkpoint reload. At the planned learning rate 0.0002, its single update
increased longest-development-request NMSE from 0.766937 to 2.790030. These are
optimizer pilot observations, not summary quality or primary campaign results.

Before any primary training, reduce the fixed learning rate tenfold to 0.00002.
Use an eight-update stability pilot with the longest training and development
requests, under a new root and a subset index referring to the immutable full
teacher cache. That subset is only a systems/optimizer check. The primary run
still uses all 64 training and 8 development requests, 96 updates per each of
14 groups, the same rank, boundary objective, checkpoint selection, seeds,
quality stop rule, and deployment controls. Retain the original pilot.

Implementation v2 added exact dense-boundary parity and prevented an attempted
reuse from writing a failure file into an existing root. Implementation v3 adds
this optimizer amendment and eight-update pilot. The completed v1 teacher
cache and all producing snapshots remain unchanged. Cache preparation did not
perform optimization and remains valid input for the v3 training producer.
