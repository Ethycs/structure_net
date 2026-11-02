# Sequential NCM decoder repair: fixed staged follow-up

The user authorized decoder repair, expansion to 732 engineering TRAIN summaries,
then sequential source-to-code training and source-disjoint semantic evaluation.
This exploratory follow-up preserves every earlier result and failed gate. It
does not reopen the prior attention-corrector campaign or its failed scaling gate.

The sequential 16-case generator produces 10,254/10,256 reference code indices;
all rendered outputs equal the imperfect reference-code decoder output. Test the
remaining bottleneck by changing only the text decoder. Encoder, codebooks,
generator, concept/source attention, word/source attention and frozen Llama source
features remain unchanged. Llama is absent during optimization. Reference codes
are training inputs, never available at source-only inference.

Each stage initializes from the same completed sequential checkpoint
`d27af43e7bb8f8a80d81e2c19c51e12ddb8bcc08154dd0540303862ecd81f841`.
The full-data decoder does not inherit the extra 16-case control training. Use
the already admitted 732 complete engineering summaries, excluding all reserved
source components. The 16 control IDs remain the prior fixed source-only selection.
No new data, held-out labels, truncation, teacher calls or target rewriting.

Run CPU routing/loss/gate tests, then a two-update CUDA pilot with first and longest
control source, then a fixed 256-update 16-case control. Scale only if oracle-code
decoding matches every active target token and all 16 rendered texts, and actual
source-only generation also renders all 16 target texts. Exact reconstruction here
tests codec identity, not semantic summarization accuracy. Original casing cannot
be reconstructed by the inherited uncased tokenizer; rendering comparisons use the
same native tokenizer for targets and predictions, without claiming literal fidelity.

The authorized full decoder stage is 1,024 updates on all 732 TRAIN cases. Before
subsequent source-to-code training, require >=99% active word-ID reconstruction,
>=95% exact rendered TRAIN summaries, and no empty summaries. These are only
representation checks; passing them cannot establish 95% task accuracy. A failed
gate stops this fixed campaign and requires a separately recorded diagnosis or
new protocol, rather than silent additional updates or checkpoint selection.

Use microbatch 1, accumulate 4 uniformly shuffled TRAIN examples without replacement
within each epoch; per-example native word cross-entropy ignores PAD0 and trains
EOS101. AdamW, betas .9/.95, epsilon 1e-8, weight decay .01, clip 1. LR linearly
warms for 32 updates to 1e-4 and cosine decays to 1e-5. FP32 weights, FP16 autocast
and gradient scaling during training; FP32 generation; TF32 off; GPU2; allocation
cap 4.9GB. Sampler seed 151142, dropout seed 152142. Fixed final checkpoints only.

Hash all protected parameters and buffers before/after, verify unchanged source-only
codes, and prove decoder-only gradient routing. Save decoder, parent identity,
optimizer, scaler, sampler state and remaining permutation, CPU/CUDA RNG. Reload
into a fresh parent and require exact tensors, optimizer/scaler/RNG and rendered
outputs. Freeze source/input hashes and source snapshots before each run, observe
actual terminal tool receipts, and preserve raw per-step losses and all outputs.

The later generator protocol must be fixed before launch. Independent semantic
judging uses full sources, supported claims, coverage and coherence; exact summary
matches are not the task metric. Keep original sealed final evaluation unopened
until development quality qualifies. Measure end-to-end speed including Llama
source processing only after the quality gate. Service readiness probes contain
synthetic JSON only and cannot qualify a semantic judge or rescue a failed gate.
