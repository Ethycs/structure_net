# TRAIN probe: append-only condition outputs

The v3 pilot exited 1 after inference when it tried to rewrite `outputs.json`
using the append-only JSON writer (session 42355, chunk d1ebb2). BROAD parity
and epoch2 repetition checks had returned successfully, but no completed
pilot result was produced. Preserve the producer, first saved condition,
failure, and actual receipt in the v3 pilot root. Do not treat it as a passed
pilot or a semantic-quality outcome.

Version 4 writes each completed condition once to its own file, then writes
the combined `outputs.json` once after all conditions. Bind every condition
file in the result. Use fresh v4 pilot/run/audit/results roots and schema
`nal.ncm-public-training-probe.v4`. The models, sixteen controls, inference,
tolerances, scientific comparisons, and active epoch4 training are unchanged.

Run the focused CPU contracts and a fresh real inference pilot before the
full diagnostic. Prior failed runs remain preserved and excluded.
