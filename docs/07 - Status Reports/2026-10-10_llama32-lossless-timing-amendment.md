# Counterbalanced timing correction

Review found that the parent timing helper's rotation followed by reversal
cancels itself for two arms: original then candidate is repeated, instead of
alternating the order. Preserve those paired-round measurements as diagnostics.
The three-arm projection diagnostic is unaffected by that particular defect.

Use a separate new timing root and an explicit original/candidate order on even
rounds, candidate/original order on odd rounds. Keep one warmup and five measured
rounds, the same three full-model workloads and 1024-token KV capacity budget,
the same decoded checkpoint, exact residual network and precision. Record the
order of every round. Use this corrected stage for the primary efficiency gate.

At registration the all-weight audit has passed, the storage costs and projection
diagnostic are known, and the complete residual pilot is running. No final
summary result or complete-model residual-versus-original primary timing has
been inspected. No quality, exactness or efficiency threshold changes.
