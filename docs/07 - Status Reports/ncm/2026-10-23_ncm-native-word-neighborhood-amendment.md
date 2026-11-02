# Native-word neighborhoods for the final NCM refinement stage

Date: 2026-10-23. EXPLORATORY amendment before the mixed-task trainer pilot or
full training has run. The earlier two-update NCM probe has unusable text despite
passing systems gates. Its final inputs are averages/residuals of native word
embeddings, which need not lie on the pretrained vocabulary manifold. This is
a plausible mismatch, not an isolated causal diagnosis or paper-replication failure.

Jacobi Forcing provides a closer causal parallel-decoding reference:
https://arxiv.org/abs/2512.14681 and https://github.com/hao-ai-lab/JacobiForcing.
Its own-trajectory/progressive-noise principle motivates preserving native word
inputs. Our NCM+attention-only adaptation is not a reproduction of that trainer,
and its published benchmark speedups are not our summarization measurements.

Keep the fixed143accepted summaries,1024language/64diagnostic windows, offline
source-KV/sparse-teacher caches, original frozen weights/books and all base
training budgets/optimizers/seeds from the neighborhood plan. The unrun soft-
embedding-only full-training configuration is deferred; no successful early-stop
claim or failed full-run outcome is inferred. Existing data/cache evidence stays
valid because source encoders and true-prefix frozen teacher remain identical.

Change the word stage: first predict all five residual concept levels, then
project the cumulative per-position estimates to nearest native vocabulary
directions by cosine similarity. Use chunked FP32 search to avoid a second dense
FP32vocabulary allocation. Exclude nonterminal special tokens; allow configured
EOS tokens, map a zero estimate to the native EOS. This is a lossy inference
proposal, not exact token recovery. Never feed current/future gold words at inference.

The word refiner uses native predecessor-token embeddings, the fixed current-
position coarse hint and unchanged causal attention. Both word arms use SOURCE-
ONLY generated/noisy concept history; target-derived concept hints are confined
to the earlier coarse-code training arms. Train half of word batches
at the true native predecessor neighborhood, with0.1RMS Gaussian perturbation
except the prompt anchor. In the other half, use source-only generated coarse
history and a native word state from0–3 actual student refinement passes. Replace
only a progressively growing suffix of the gold training state: width8,16,32,64,
128over successive fifths of post-warmup updates, relative to each valid target
length. This is training teacher forcing/consistency supervision, not an inference
oracle. Source-cache contents remain target-free. Compute the same hard/sparse-
teacher objective. Multiply the native-center word loss by10 to preserve the
pretrained language path, following the AR-loss emphasis in the released CLLM
trainer (our loss and adapter restriction otherwise differ). Metadata records
the arm, suffix width, unscaled objective and sampled rollout.

Primary inference is FIXED: five concept passes plus four native-word parallel
refinement passes (nine total), initialized from the rounded concept plan. No AR
fallback, external teacher, verifier or convergence-until-good stopping is allowed
to pass the primary quality gate. Each word pass uses only the previous generated
word state and source-derived fixed coarse hint. Return first configured EOS as
normal generation termination and retain all128raw IDs. Native causal-center or
longer-refinement diagnostics cannot rescue a failed nine-pass task endpoint.

Additional CPU contracts: vocabulary projection/ties/zero cases; current/future
word-state changes cannot affect earlier logits; native-center initialization
matches the original native teacher; noise-window boundaries and deterministic
state generation. The mixed-task GPU pilot must exercise both word arms, both
frames, full128activation sizes, frozen/source-cache checks and exact checkpoint/
generator reload before the fixed2048-update run. Add this amendment/new source
hashes to the trainer producer. All64development cases and the original>=61/64
Terra then Sonnet gates remain; final source-disjoint/full-range evaluation is
still required before completion. No task accuracy or speedup is established.
