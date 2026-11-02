# Resource-scaled NCM training configuration

Date: 2026-10-22, before any new adapted-model outcomes. Addendum to
`ncm-paper-regime-recovery-v1`. Public teacher generation is in progress; the
released native model's first development outputs have been inspected and look
poor. No new adapted-model quality outcome is available.

Start again from released XSum weights, not the previous memory/hybrid models.
Reuse a released codec only if the declared reconstruction and oracle semantic
controls pass. If that gate fails, tokenizer adaptation gets its own frozen
implementation/configuration before launch. Always pair generator checkpoints
with an explicit codec path and SHA; rebuild all generator labels after a codec
change. The accepted codec is frozen throughout generator training.

Generator adaptation: 512 unpaired updates on complete 47-token windows plus
EOS from public training passages; no development/test text is used. Prefix is
EOS followed by PAD. Then run native-prefix and frozen-Llama-context arms from
that same unpaired checkpoint. The hybrid gets 512 bridge-only warmup updates
with the generator frozen; both arms then get 2,048 conditional updates. Bridge
is LayerNorm(3072) then Linear(3072,768), newly initialized, with valid Llama
states repacked to the beginning of the 512-position prefix. Llama weights and
MLPs remain frozen; this is a new output generator after the full 28-layer body.
The native arm has no Llama body or bridge, so total compute is not matched.
These arms diagnose the interface; they are not a throughput comparison.

Microbatch 8, accumulation 2, effective batch 16. AdamW beta=(.9,.95), WD=.1,
gradient clip1. Generator LR1e-4, bridge LR3e-4, linear warmup128 then cosine to
10% of peak over each stage (bridge warmup32). Published generator dropout .3,
BF16 autocast with FP32 parameters, objectives and nearest-code assignment.
Uniform same-level conditioning replacement decreases linearly .30 to zero;
labels remain clean. Use alpha0 (equal code-position weights), an explicit
choice allowed by the paper. The exact XSum corruption setting is not released.

Seeds: model42, dropout52, minibatches2042, augmentation3042. Native and hybrid
conditional stages receive identical sampled indices and augmentation seeds;
bridge warmup has separate streams. Shared unpaired training is not an
independent replicate. Save optimizer, all RNG states, stage/step, codec SHA,
levels and vocabularies. A fresh two-update lifecycle pilot covers unpaired,
bridge-only and joint updates, query/bridge gradients, frozen state hashes and
exact checkpoint reload. It also measures batch-8 memory at one GPU worker.

Save development outputs at conditional steps512/1024/1536/2048, plus fixed
32-case train diagnostics at the final step. These checkpoints are development
iterations, not independent tests. Grade with the unchanged single-candidate
rubric. Final-test admission needs >=61/64 development passes (>=95%), complete
valid judgments, and passing teacher/codec controls. Development misses remain
visible and motivate the next new study. The 256-case sealed test remains
unopened until admission; final success still requires >=244/256 under both
qualified judges. No automatic CPU fallback, no service changes, no speed claim.
