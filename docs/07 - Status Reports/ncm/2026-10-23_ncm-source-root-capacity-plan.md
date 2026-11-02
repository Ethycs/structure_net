# Isolate source-to-root concept learning before another complete NCM run

Date:2026-10-23. Exploratory hypothesis
`ncm-word-final-source-root-capacity-v1`. The preceding word-final model reached
0/64short-development passes with64valid judgments; its longer cohort is still
being graded. On16fixed TRAINcases, correct concepts recovered1567/1635words,
but source-generated concepts recovered21/1635. Its root level recovered only
6/176active-block labels. Generation and full forward at generated concepts
agreed on3067/3072word IDs. These are separate task and component measurements.

Do not launch this training until the complete preceding128-case evaluation,
raw failures, mechanism diagnostic and conservative registry readback are
archived. Preserve every previous failed run. If the preceding development
gates pass, this failure-driven capacity experiment is not admitted.

Test whether the source-conditioned root classifier can fit16TRAINsources when
trained directly, without competing word losses. Reuse the exact16cases and
length-stratified hash selection from the generation-gap diagnostic. Every case
remains TRAIN. No development text, judge critique or final cohort enters this
training. This overfit/capacity check cannot count as model task accuracy.

Initialize from the fixed final512word-final checkpoint, never an intermediate
or pilot. Keep original Llama weights/MLPs and codec frozen. The original parent
is absent during optimization. Freeze the copied word projection/decoder and
all concept heads except level0. Train the new source bridge and generator
shared transformer, root queries and root head. These are new module copies;
no original parent weight is trainable. Retain generator training dropout.3.

Compute only source embeddings plus16root queries through the generator, using
the same complete source features, masks and device-safe geometry. No target
concept or word history enters prediction. Root labels enter cross-entropy
only. A block is active if its48word targets contain any non-PAD token; this
mask is used in loss and component measurement only. Optimize mean root CE on
active blocks plus.25times mean CE on entirely padded blocks. This prevents
easy padded blocks from dominating the16-label root loss. The contrast with the
previous run changes gradient scope, loss weighting, learning rate and data
size; it is a capacity diagnostic, not an isolated causal attribution.

First run a fresh two-update real CUDA pilot on first and longest selected
TRAINsources. Pad the latter to the largest observed nonfinal frame2816for
memory coverage. Require finite losses and gradients in the root/source path,
no gradients or changes in frozen words/other heads/codec, peak allocation
below8GB, and exact model/optimizer/CPU-CUDA/sampler RNG and source-only reload.
The pilot weights are discarded. Use physicalGPU0after earlier workers exit.

Then start fresh from final512with exactly256updates, batch1, uniform sampling
over16cases. Model/dropout seed13143; sampler seed14143. FP32master/AdamW,
BF16autocast, FP32CE, TF32off. AdamW beta=(.9,.95),wd.01,clip1;16warmup updates
to3e-4then cosine to3e-5at256. Save128for recovery; select only final256.
No noisy histories, teacher logits, adaptive extension or checkpoint search.

Measure root argmax IDs on all16TRAINsources before and after. Also score each
case's labels against the root IDs produced from the next case's complete
source in a fixed cyclic order. Retain all IDs and counts. A capacity pass
requires at least95%active-root recovery,95%all-root recovery, and an active
recovery drop of at least50percentage points when sources are shuffled. These
are deliberately TRAIN-only source-dependence gates, not held-out semantics.
Preserve failures even if source-only text later looks plausible.

Use fresh roots `20261023_word_final_root_capacity_{pilot,train,results}_v1`
under the paper-regime directory. Bind source, plan, final512checkpoint,
original128-case evaluation and registry identity, fixed training labels and
feature caches. Record actual terminal receipts and replay sampling/metrics.
A pass permits a separately frozen full-hierarchy/source-only training protocol;
it cannot establish95%complete-summary task accuracy, compression or speed.
The final task objective and both closed final cohorts remain unchanged.
