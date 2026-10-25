# Bound repair-head scores before longer training

The engineering v1 two-update lifecycle actually completed and strictly reloaded
all state, but exposed code CE around669,674 and367,867 while word loss stayed
flat. Preserve its weights, trace and unchanged baseline. Do not launch v1 long
training on that scale. Frozen book analysis found root-vector norms ranging
from effectively zero to7,221.8, with generated roots near2,085median and many
target roots effectively zero. Other levels have much smaller norms. EMA counts
are positive for all codes, so these statistics do not justify dropping IDs.
The native vendor uses raw code vectors; do not claim it normalizes them.

In new v2 mechanism and run roots, precondition classifier vectors with
phi(v)=v/(1+norm(v)), then constrain learned query vectors to the unit ball.
Score all original native IDs with -4 times squared distance plus the unchanged
positive copy preference. Scores are bounded; the vocabulary, raw native books,
encoder, generator, hard code-to-latent path and frozen word decoder are intact.
Zero residual still preserves every native input ID, including duplicate-like
zero-vector entries. This is a new classifier geometry, not changed quantization
or a paper replication claim. Exact code-ID agreement is not semantic accuracy.

Reuse the completed v1 dataset of732engineering TRAIN source/target/actual-error
bindings. No historical atom targets are admitted. Use the same full hierarchy,
same corrector parameter count, seeds, task-word straight-through objective,
two-update lifecycle,512update16caseTRAINcontrol and4,096updatefullrun specified
in the v1 engineering plan. Start fresh at identity; no v1 pilot weights reused.
Run CPU identity/source/gradient/mask tests including extreme-norm books, then a
real v2 CUDA lifecycle and strict reload before longer training. Preserve v1's
numerical concern as a failure of raw-distance conditioning, not model quality.

All parent/MLP and original NCM tensors remain frozen. The original128DEV and
sealed320final evaluation scope, plus156additionalengineeringtestinputs, remain
unchanged. No semantic95claim, speedup, clipping or target repair is authorized
by passing this numerical lifecycle. Semantic services still need readiness and
the completed3,539-first lineage before any new judgment requests.
