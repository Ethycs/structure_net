# Native concepts and a parallel native-word readout

Date: 2026-10-23. EXPLORATORY `ncm-native-concept-interface-v1`.
This is a new NCM adaptation, not an unchanged reproduction of the paper's
learned VQ-VAE. The original NCM text attachment remains0/64; copying37/64;
native parent49/64. The native-word AR capacity control is still running.
Its intermediate teacher-forced loss is not a task-quality endpoint. The
new128-token teacher-label pilot is stopped on provider quota, with no labels.

The proposed generator retains NCM's discrete multiscale residual concepts,
coarse-to-fine conditional prediction, parallel positions within each level,
and corrupted-history training. It changes the final readout: native Llama
vocabulary prediction replaces the learned text decoder. Do not call an AR
head, supplied final word IDs, reconstruction arithmetic or a baseline passing
the95%NCM success result.

## Representation control before a student run

Use frozen native word embeddings in FP32; no learned text encoder or decoder.
Fit residual codebooks at lengths4,8,16,32,64 for a128-token canvas,256 entries
each. Pool each residual over equal contiguous cells by an arithmetic mean;
expand chosen vectors by repeating within their cells. Reserve entry0 as the
exact zero vector, never updated, so adding a chosen code cannot increase
squared reconstruction error relative to adding nothing at that same history.
The final language stage predicts actual native token IDs, including EOS;
coarse quantization is conditioning, not a lossy decoder of final words.

Representation data are512 prospectively hash-selected public TRAIN sources,
plus a disjoint64-source diagnostic, from the existing8,192-source public
training pool. Order is SHA256("native-concept-bank-v1:"+id). Keep one original
group/session/source hash each. Tokenize the first128 native tokens, padding
shorter raw passages with native EOS. This truncation is explicit representation
window preparation, never task-source truncation or evaluation exclusion.
No summary labels, development cases or final-test outputs enter codebook fit.

Use native embedding tensor from the verified original Llama snapshot. FP32 on
physicalGPU2, disable TF32, seed42. Each codebook initializes entry0 to zero and
the other255 entries from distinct pooled training-vector indices using a
separate deterministic CPU generator seeded3042+level. K-means runs32 complete
assignment/mean iterations, keeping empty centers unchanged. GPU floating-point
index-add reductions need not be bitwise repeatable; freeze actual resulting
centers and preserve this limitation rather than claiming independent repeats.
Distance comparisons and assignments use FP32 with chunks of at most1,024.

Before fit, focused tests cover zero-code identity, non-increasing squared error,
clean-history/static-label parity, conditional repair after wrong earlier codes,
padding/cell boundaries and save/reload. A real GPU2 pilot uses64 sources, two
iterations per book, otherwise identical shapes/dictionary, and both clean and
corrupted-history checks. Require at least3GB free before loading the FP32 word
table. The full fit requires a matching passed terminal pilot, stable producer
and complete512/64 source coverage. Stop on OOM, nonfinite values, model/source
drift, numerical invariant failure or inconsistent saved centers/assignments.
Preserve failures in their original roots. Report per-level MSE and active
entries on both cohorts; neither is summary accuracy or a quality gate.

Roots `20261023_native_concepts_{prepare,pilot,fit}_v1`. Save raw ID windows,
centers, assignments, runtime/config/source identities and explicit scope.
No student-training admission follows from low embedding error alone.

## Later generator interface and gates

The implementation may be prepared while the AR capacity run is live, without
editing its producer or touching its GPU. Source memory is the full per-token
native-body feature sequence. For each concept level the five-block frozen
native readout receives the resized cumulative earlier reconstruction plus a
level query; a small trainable head predicts that level's discrete codes.
All current-level positions attend bidirectionally within their block and to
source memory, without seeing current/future labels or other level blocks.
The128-position final block predicts native words through the frozen tied LM
head. Native base/FC parameters stay frozen. Inference makes no full-parent
generation/verifier call after source feature extraction.

Training uses real corrupted earlier histories; conditional residual experts
are recomputed against those histories. Native final word labels remain clean.
Root queries and prior-level inputs must have no current-label leakage.
Prefix-only KV caching must match a full source-plus-current-level forward
within declared precision tolerance and retain the original source cache.
Tests must prove mask isolation, gradients, frozen word identity and immutable
source caches. A hierarchy ablation remains necessary to determine whether
the NCM mechanism contributes rather than being ignored by the native readout.

A separate declared generator-training protocol is required after the AR
capacity endpoint and representation pilot are complete. Small32-example
mechanism controls may use the already accepted32 labels; broad student runs
still require the independent new teacher-label quality gate and a prospective
source-disjoint task protocol. Final task coverage must include the application's
128 stored tokens and full input range, not silently inherit the prototype's
short-source ceiling. Final256 cases stay sealed until the relevant gates pass.
The95%complete-summary objective and measured speed comparison remain intact.
