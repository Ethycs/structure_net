# Experiment 03 — Qwen Teacher Audit

**Verdict:** COMPLETE UNDER USER-AUTHORIZED AMENDMENT  
**Runtime:** canonical NAL CPU/API campaign  
**Date:** 2026-08-16

The canonical hash-selected audit contains 2,048 unique eligible PAWS training
groups, balanced 1,024/1,024 by human label. Qwen returned an exact legal label
for every request with zero terminal failures and achieved 78.91% accuracy
against the human labels. The selection digest is
`f00d9651545754041d9540337d78c533161ba139747f4acd99e4ddd6d9668b72`.

The earlier all-train annotation attempt reached 18,752 responses before the
user correctly observed that PAWS already supplies authoritative answers. It
is preserved as `annotations_superseded_full_attempt.jsonl` and excluded from
the canonical auxiliary-loss lookup except where a record belongs to the
frozen 2,048-group audit. The canonical file, summary, and NAL ledger are under
`data/experiments/paws_abc_routing/2026-08-16_experiment_03/`.

Experiments 04–05 are licensed to train on all 49,259 eligible unique human-
labeled groups. Qwen labels may contribute only the preregistered 25% auxiliary
loss on canonical audit rows.
