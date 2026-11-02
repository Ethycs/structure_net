# Preserve quality evaluation while removing repeated static zero checks

Date: 2026-10-23. Clarification for the isolated native quality-teacher study;
the 252-label training protocol, 512-update budget and all quality gates stay
unchanged. The current measured NCM/hybrid accuracy remains 41/64.

After the final training worker is observed terminal, use the first two fixed
development sources for the inference/reload pilot. Retain the first plain
implementation outputs as the canonical candidates. Generate the same complete
128-update word histories with an inference-only static adapter runtime and
require exact equality, including every word state and final decoded text.
Also compare the initial full-vocabulary logits exactly. Reload deliberately
perturbed word parameters and require the same complete histories again.
Only after this terminal-success pilot generate the remaining62sources and
reuse the canonical first two unchanged.

The static runtime computes whether each word B matrix is zero once when
installed, rather than doing a GPU reduction/synchronization in every layer
of every word iteration. It performs the same base and low-rank linear
operations. It rejects gradient-enabled calls and changed parameter versions,
restores original methods on exit, and changes no weights or parameter names.
CPU tests cover zero/nonzero output equality, absence of repeated reductions,
changed-weight rejection and method restoration. Real GPU equivalence is
mandatory before full generation. No timing or tokens/s claim follows.

This remains an unoptimized native-language quality teacher with128whole-frame
updates plus a fixed-point verification pass, using no concept hint/bias or
external verifier. It is not declared a working NCM replacement. Grade all64
frozen summaries under the qualified128-token rubric, preserving all256prior
first judgments and their negatives. Terra must reach64valid/61passes before
independent Sonnet on identical texts. Both final cohorts remain sealed.
