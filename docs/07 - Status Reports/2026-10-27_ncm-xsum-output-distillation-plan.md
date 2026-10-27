# Train the prefix bridge through the frozen NCM's output

Exploratory hypothesis `ncm-xsum-output-distillation-v1`: matching final code
features, followed by code-distribution distillation, preserves the released
conditional XSum NCM better than equally long additional input-MSE training.
The previous bridge failed: native-content agreement 16.17%, projected MSE
50.39% of the constant-mean control. Those outcomes are known. This is a new
objective comparison, not a retry or a semantic-accuracy claim.

Motivation: [Mai et al., model stitching](https://arxiv.org/html/2603.12433v1)
found final-feature matching useful for frozen vision models. Its application
to NCM code generation is an unverified extrapolation. Keep the main public
curriculum on GPU1 unchanged. Use GPU0 only, locked Pixi ncm, FP32, TF32 off,
deterministic algorithms and CUBLAS :4096:8, two OMP/MKL threads, allocated
memory <=4.9GB. No other GPU's workload is changed.

Reuse exactly the prior 512 bridge-fit sources, 128 bridge-probe sources,
fit-only normalization and final bridge checkpoint. All sources fit the native
512-token prefix without truncation; outputs retain the native 48-position
frame. Probe outputs have already been inspected in the parent experiment:
this is a reused development probe, not a new final test. Public TRAIN sources
may have been seen by the released teacher. Application DEV and final cases
remain excluded. No new LLM labels, judges, summary targets or speed tests.

Only the same 3,540,352-parameter 3072->1024->384 bridge is trainable. Freeze
the Llama body, released codec and generator. For each fit source, generate
the released native code trajectory once, then obtain all 200 final code
features from a full block-diagonal teacher-forced forward. Save FP32 features
and integer codes, with source and file hashes. Reconstruct teacher logits
from those features through the frozen per-level heads. Do not cache the much
larger logit arrays. The student receives predicted prefix embeddings and
the identical previous-level code history; source token IDs are never student
inputs. Generation uses only dummy zero content IDs with the original padding.

Two arms start from the identical completed input-MSE bridge:

- `mse`: 10 further epochs of the original normalized projected-prefix MSE.
- `distill`: 5 epochs matching final pre-logit code features, then 5 epochs
  minimizing teacher-to-student KL at temperature 1. Feature MSE and KL each
  average within a hierarchy level and then equally across all nine levels.
  Targets are detached. No code-history corruption is added in this comparison.

Each arm has 5,120 updates, one full source/update. Use identical shuffled
source orders from an independent CPU generator seeded 6519. AdamW betas
0.9/0.999, eps 1e-8, weight decay .01, foreach false, gradient clipping 1.
Cosine LR 1e-4 to 1e-5 across each arm, without a phase-boundary reset.
Models stay in eval mode to disable pretrained dropout while gradients pass
through the frozen generator to the bridge. Each arm saves complete model and
optimizer checkpoints every 512 updates. Evaluate only the fixed final state.

CPU tests cover loss direction, detached targets, equal level weighting,
phase/learning-rate boundaries, disjoint fixed input identities, strict
checkpoint state comparison, cleanup of feature hooks and conservative gates.
The CUDA pilot uses the first and longest fit sources. Require native full
forward code decisions to equal native generation on both, exact equivalence
of native and identical replacement prefix features/logits, nonzero finite
bridge gradients, no frozen gradients, and exact save/reload continuation
across feature and KL updates. Also exercise the MSE control. Replay all heads
from final features and require exact logits. The cache stage records full
forward versus generation decisions for every fit case; >=99.9% agreement is
required before training. This permits only a measured tiny numerical
full-forward/level-forward difference; the pilot's exact gate is unchanged.

After both complete arms, generate all 128 probe outputs. Reuse the parent's
native and starting-bridge outputs, verifying the native codes and words anew
on the first and longest probe. Save both new arms' complete codes and word
IDs. Primary preservation gate: distill >=90% agreement with native on all
48 decoded positions, non-padding positions, and content before first EOS;
native content must be nonempty on every case. For the directional hypothesis,
also require distill content agreement >=5 percentage points above the matched
continued-MSE control. Report code agreement and paired per-case differences.
Exact agreement is a mechanism diagnostic: different wording may be valid.
No outcome here establishes semantic summary accuracy or the 95% goal.

CPU audit replays every update ID/phase and all raw code/token counts, verifies
source/checkpoint/frozen-model hashes and actual process receipts, then writes
the measured report and conservative registry entry with independent exact,
semantic and JSON read-back. Failed stages and prior failures remain intact.
Any changed producing code after a GPU stage starts requires a new version and
root. Fresh roots use `20261027_xsum_output_distill_<stage>_v1` under
`data/experiments/ncm_paper_regime`. Long training uses a durable systemd user
service. A launch receipt alone never establishes training completion.
