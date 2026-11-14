# All-block Qwen: numerical-rank correction and five-point accuracy budget

Prospective amendment for `JSPACE-QWEN-ALLBLOCKS-01`, dated 2026-11-14 UTC.
It preserves the first attempt under `jspace_qwen_parallel/20261114_v1` and
authorizes a separate, fingerprinted `20261114_v2` producing version.

## What was known before this amendment

The 58-test implementation suite, 36-block shape-equivalent GPU backward and
packed-validation pilot, exact fused/reference check, cache/reload lifecycle,
three-GPU fitting pilot, six-block joint recovery lifecycle, and next-teacher
gradient capture passed. The systems-only packed layout used 3,816,752,960
resident tensor bytes including the regenerated rotary buffer, a 38.159%
reduction. Those systems checks are not task-quality evidence.

The first primary attempt completed the full group-0 capture (576 cases,
128 Jacobian calibration cases, four task families), all six basis/interface
fits, and polynomial fits for layers 31, 32, 34 and 35. Layer 30 was rejected
after its four streamed Jacobian passes. Layer 33 was not launched; no primary
recovery or held-out scoring ran. All artifacts and failed logs are retained.

The original numerical-rank guard compares singular values with
`sigma_max * max(original_rows, channels) * eps`. Its relative threshold grew
from approximately 0.001312 to 0.033007 when the full-context bank grew to
276,880 rows, even though the SVD itself is performed on an 80 by 11,008 sketch.
A CPU counterexample confirms that duplicating rows can change admission under
this guard while leaving the singular-value ratios unchanged. Successful
layers are not sufficient evidence about the rejected layer's spectrum.

## Numerical change, specified before inspecting the rejected spectrum

Use `sigma_max * max(actual_svd_shape) * eps` for the numerical rank of the
matrix whose SVD is actually computed. Require at least 24 singular directions
above that threshold, preserving the fixed 24-anchor budget. Record the old
original-row cutoff, its rank, the new cutoff and rank, all 80 singular values,
and pivot norms. A genuinely deficient sketch still stops the run. This is a
numerical-independence test; it does not establish activation reconstruction
accuracy, retained information, or final task competence.

Do not discard rows, filter zero rows, change row weights, normalize examples,
change the pooled activation scale, alter random draws, or modify the QR/SVD,
pivots, coefficient fitting, channel-removal budget or polynomial metrics.
The guard-only candidate has CPU tests demonstrating exact equality with the
old coefficients, anchors, removals and metrics when the old guard admits a
fit, duplicate-row admission stability, and rejection of actual rank deficiency.

Before launch, the read-only layer-30 diagnostic must finish, verify its input
and source hashes, and show at least 24 directions under this predetermined
sketch-sized rule. It also reports row/case/family energy concentration so a
dominant outlier is visible. The first diagnostic attempt failed before any
spectrum calculation because CUDA memory statistics were requested before
initialization; its producer is preserved. The corrected diagnostic changes
initialization only and has a separate root.

## User-authorized quality budget

The user explicitly confirmed allowing **up to five percentage points of final
task accuracy loss**. Set the primary aggregate accuracy-drop limit to 0.05,
relative to the original native model on the same fresh 512 examples. This is
an absolute percentage-point budget, not a five-percent relative reduction.
Keep the primary size and whole-model throughput targets at 38% and 1.4×.

Report each task's accuracy, paired wins/losses, and the original one-point
aggregate comparison. The original per-family threshold remains a diagnostic;
it no longer vetoes the user-authorized aggregate five-point trade-off. The
budget applies to final evaluated accuracy, not to training loss or the
singular-value cutoff. Preserve exact packed reload and graph/static execution
validity checks. The result remains exploratory and `confirmed=false`.

## Reuse and producing-version boundary

Reuse the unmodified task identities, native model, capture, bases, interfaces
and four already successful fits from v1. Hardlink immutable tensor/array files
to avoid duplicating the calibration bank; preserve original absolute input
references and all original source fingerprints. Every imported stage receives
a wrapper manifest identifying its originating root, fingerprint and manifest
hash. Imported outcomes are explicitly identified as produced by v1.

Validate the recorded live and archived v1 source bytes, all source/input and
artifact hashes, and absence of primary training/held-out scoring before
import. The new plan archives its complete v2 source and includes the amendment
and diagnostic evidence in its fingerprint. Only missing layers 30 and 33 need
new group-0 fits. Later groups use the revised numerical gate. All model sizes,
training settings, staged teacher selection, split sizes, seeds and runtime
measurements remain as specified in the original plan.

The untouched final evaluation pool is reused because no v1 held-out outcomes
were observed. Report both the stopped numerical-gate attempt and the revised
campaign, including any subsequent failures. Do not pool their completion
counts or count the six sequential groups as independent replications.

## Diagnostic result before v2 launch

The corrected read-only diagnostic completed under fingerprint
`0a8831e1d0d45c63262e66edc580ab124d6292cb4d821ea4b198ef02a96044d3` at
`data/experiments/jspace_rank_diagnostic/20261114_layer30_v2`.
The old cutoff admitted rank 3; the predetermined sketch-sized cutoff admits
rank 80. The 24th and 64th singular values are respectively 0.007109 and
0.004202 times the largest, above the sketch-relative cutoff of 0.001312.
The top 64 sketch components retain 99.316% of measured standardized
Jacobian energy.

The largest case contributes 1.553% of total energy, and the four families
contribute 21.297%–31.354%; no individual outlier drives the spectrum. This is
a concentrated measured spectrum, not evidence that its directions already
have identified semantic meanings. All rows and their original weights are
retained. These diagnostics satisfy the prospective condition for the
guard-only correction; task accuracy remains unmeasured in the primary run.
