# Llama 3B output cardinality correction

Date: 2026-10-13. Parent `llama32-3b-summary-throughput-v1`.
No held-out outputs have been generated. Prompt v2 recovered final facts but
produced extra and repeated fragments on single-fragment development inputs.
Those outputs fail, including when every required string appears somewhere.

Prompt v3 states the number of source fragments and their labels explicitly.
It requests exactly that many summary lines, then stopping, with the same
128-token per-fragment ceiling. Every source byte, gold fact, seed, generation
cap and EOS rule is preserved. Apply the same wrapper to all model arms.
Reject any prepared prompt exceeding 6,300 tokens; never truncate evidence.
Keep v1 and v2 failures. Validate on development before sealing held-out runs.

Preparation note: the first verbose v3 wrapper exceeded the raw prompt limit;
the guard rejected it before writing fixtures. An inadvertently started runner
then stopped before model loading because that fixture file did not exist. The
wrapper was shortened, preserving its requirements and every source byte.
