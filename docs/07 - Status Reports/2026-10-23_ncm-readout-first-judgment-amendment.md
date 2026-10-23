# Reuse first judgments for identical readout packets

Date: 2026-10-23, before main readout grading. The four-case judge systems pilot
returned four valid judgments, one semantic pass and three failures. Full
selection is complete; no main condition's64-case accuracy has been measured.
Preserve every pilot judgment, including its negative outcomes.

The original generic grader would call the API again for pilot cases and for
identical source-summary packets occurring under different selectors. To
enforce first-valid outcome preservation and avoid redundant requests, use a
fresh evaluator version/root with a cache keyed by the **entire exact rubric
packet and judge role**. Seed the secondary cache with the four pilot rows
after raw parser replay. Reuse positive, negative and exhausted-invalid rows
uniformly; never select the better result. Different text, source, case ID or
judge role must have a different key. Do not reuse Terra outcomes for Sonnet.

Grade only previously unseen packets with the same four-worker, one-candidate
requests and existing maximum-three transport/schema attempts. Keep the same
64 cases per non-oracle condition,20 reference diagnostics,128-token cap,
rubric,61/64 ordered secondary/primary gates and19/20 reference threshold.
Retain full per-condition rows and denominator64 even when an API judgment is
shared. Record every cache hit, original raw attempt path, first source root,
unique request count and full packet identity. Verify reused decisions against
raw first-valid responses. No model, selection, data, score or threshold change.

The original helper/source, selection producers and pilot remain immutable.
Use `ncm_source_readout_grade.py` and a new
`20261023_source_readout_grading_v2` root, preserving both evaluator sources and
this amendment in the producer. CPU tests require role-sensitive exact packet
identity and preservation of negative/invalid rows. This is a conservative
evaluation/provenance correction before primary outcomes, not a rerun to rescue
a failed semantic result. No NCM training, final-test opening or speed claim.
