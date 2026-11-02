# Can the frozen NCM codec correct an imperfect generated prefix?

Exploratory diagnostic `ncm-residual-recoding-probe-v1`, written before this
probe runs. The broader generator has finished 4,096 updates; all 128 DEV
summaries are undergoing the unchanged semantic evaluation. Four pilot
judgments were valid and failed quality. This diagnostic does not alter those
outputs, judgments, training, or the 95% complete-summary goal.

Existing all-1,376 TRAIN diagnostics show code agreement of 100% at level 0,
99.58% at level 1, and 79.12% at level 2 during rollout. At level 11, agreement
is 95.85% with correct histories and 36.04% with generated histories. These are
exact code diagnostics, not semantic accuracy.

Infinity section 3.4 recomputes later residual targets after perturbing earlier
codes: <https://arxiv.org/html/2412.04431v2>. Its visual bitwise quantizer differs
from our text codebook. Applying residual retargeting here is an untested
analogy. Our current corruption retains clean targets. Before training a new
objective, test whether the existing later codebooks can represent corrections
to actual generated-prefix errors at all.

Use exactly the 16 TRAIN controls fixed before the broader run, with the same
full target frames and already recorded source-only generated codes. Keep the
first 0, 2, 3, 4, 6, 9, 11, or 12 levels of each generated hierarchy. Compare:

1. Append the original clean target's suffix codes.
2. Re-encode the remaining residual against the clean target's encoder latent,
   subtracting the forced prefix contributions before quantizing the suffix.

Decode both with the frozen bare codec and with its frozen word/source
attention. Preserve all raw IDs, codes, target bindings, texts, active-token
counts, exact active-frame reconstruction, and latent squared error. Compare
matched cases at every prefix length; no outcome-based example selection.
Targets are intentionally supplied to this oracle diagnostic. Neither arm is
a source-only summarizer, and no accuracy or deployable correction claim can
be inferred from it. No optimizer update, new teacher/judge request, DEV source
access, or final-test access is permitted.

Validate that zero forced levels exactly reproduce native encoding; every
clean forced prefix must recover the original full hierarchy. Twelve forced
levels must reproduce the saved generated hierarchy and its source-bound raw
decoder output. At zero and twelve levels, the two arms must agree. Use a
two-case CUDA pilot with exact repeated outputs, then all 16 cases. Verify
whole-model tensor identity before/after and exact pilot/full agreement.
Focused synthetic tests must catch target/prefix confusion and demonstrate a
case where adaptive residuals correct an error that the original suffix cannot.

Run from repository root with locked Pixi ncm, physical GPU1 RTX2060, FP32,
TF32 off, no target/source clipping, and the existing 256 MiB disk-feature LRU.
Record actual peak allocation and fail above 4.9 GB. New append-only roots are
`20261025_residual_recoding_probe_{pilot,full}_v1` under
`data/experiments/ncm_paper_regime`. Fail closed on provenance, parity, numeric,
or lifecycle violations. No quality cutoff admits a new training study: inspect
the complete diagnostic and then specify any training intervention separately.

## Lifecycle amendment, before any probe output

The v1 pilot exited with `KeyError: path` while building its input manifest,
before diagnostic inference or output creation. It incorrectly treated the
streaming feature manifest as a single-path file. Preserve its failure receipt
under the v1 pilot root. Revision v2 binds and verifies the source metadata,
feature header, and compressed payload separately before model construction,
with a representative manifest test. Use fresh `_pilot_v2` and `_full_v2` roots.
All examples, comparisons, model tensors, numerical settings, and interpretation
remain as specified above; no diagnostic quality outcome was available.
