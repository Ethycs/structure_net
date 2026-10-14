# Tensorized initialization followed by neighbourhood distillation

Date: 2026-10-13. Status: prospective EXPLORATORY, UNDERPOWERED (one trained
model). Hypothesis: `llama32-3b-neighbourhood-tensorization-v1`.

Question: can rank-1024 factorized MLPs, initialized from the original Llama
3.2 3B Instruct and trained in independent two-layer neighbourhoods, retain
bounded-summary quality while improving actual inference performance?

The teacher is the pinned original checkpoint in the cotenancy model manifest,
converted to FP16 arithmetic. All 28 layers, attention, embeddings, norms and
ordinary residual connections remain. Each gate/up/down matrix becomes two
trainable factors; there is no dense correction operator. Initialization uses
input-channel RMS scaling, randomized SVD (16 oversamples, two iterations),
and balanced square-root singular values. This is diagonal activation scaling,
not the full covariance whitening algorithm of SVD-LLM. Factors use FP32 master
weights during training and FP16 during deployment.

Train 14 independent two-layer neighbourhoods on cached original-teacher
boundary activations. Only factors train. AdamW, learning rate 0.0002, zero
weight decay, norm clip 1, 96 updates/group, minibatch one request, seed 31013;
data-order seeds and decomposition seeds are derived separately. Select the
lowest development boundary MSE checkpoint among updates 0, 32, 64, 96. Cache
prompt plus reference-summary tokens so supervision includes answer positions.
Local training always receives original-teacher inputs; assembled evaluation
must therefore test accumulation of errors across neighbourhoods.

Data: new synthetic final-decision dialogues, disjoint seeds 21000–21063
(64 train), 22000–22007 (8 development), 23000–23019 (20 held out). Four
workload shapes cycle: approximately 500/2000 input tokens, eight fragments
near 6300, and hierarchy around 1000. Templates are inherited from the previous
summary study; seed separation does not make these representative production
data. No previous held-out request is used for training. Sources stay below
2048 tokens/fragment and 6300 tokens/prompt. Output caps remain raw 768,
hierarchy 512, with at most 128 tokens per stored summary. Do not truncate
reference answers to meet the input prompt limit; appended answers are training
positions, not additional source fragments.

Controls: same-backend original FP16, untrained factorized initialization,
distilled factors; and the existing pinned native Q4_K_M + lookup-8 deployment
baseline on exactly the same evaluation inputs. PyTorch prototype versus native
Q4 is an actual implementation comparison, not an isolated architectural
comparison. Parameter count and checkpoint bytes are distinct from memory or
speed. No cross-request KV reuse. Per-request lookup may use only that request.

Decision order:

1. CPU tests, tiny real CUDA training/save-reload, then full-architecture
   memory pilot (including the longest training request), serial on physical
   GPU 0. Finiteness, teacher frozen, all factors receive gradients, reloaded
   output agrees. Systems checks are not quality evidence.
2. Train the fixed full 14-group student. Record initial/selected development
   errors, all update histories and parameter counts, even if unsuccessful.
3. Evaluate all three PyTorch arms on the eight development requests. Source
   review requires correct current facts, no superseded facts, no invented
   facts or fragments, and length/cardinality compliance. Training selection
   uses boundary error only. If the distilled model fails any development
   request, stop promotion: report the failed quality gate, preserve outputs,
   run a bounded matched runtime diagnostic and the native Q4 control, and
   leave the held-out quality stage unexecuted. This is a legitimate completed
   exploratory failure, not a claim that all tensorization methods fail.
4. Conditional on all eight development requests passing, evaluate the sealed
   20 held-out requests. Require >=95% source-reviewed complete summaries and
   no lower pass count than original FP16 or Q4. Five timing repetitions of
   fresh requests; require >=2x output tokens/s over Q4+lookup and lower mean
   request latency for the user's speed target. Any lesser positive speedup
   is reported separately. No native export/kernel project is authorized by
   a local MSE improvement alone; pursue that stage only if quality survives.

Always report local error reduction, whole-model FP16 parameter bytes, actual
timing boundaries, and quality separately. Local MSE is never task accuracy.
Failures/OOMs are retained; implementation retries need a new version/root,
not replacement of evidence. No threshold-miss retries. Unexecuted conditional
stages must be explicit. This study cannot establish 95% production accuracy
or mobile performance.

Artifacts: `data/experiments/llama_neighbourhood_tensorization/20261013_*`.
Content-hash/snapshot code, this plan, fixtures, teacher manifest/checkpoint,
training cache, factor checkpoints, environment and evaluation producers.
Use strict JSON and independent final artifact readback, a dated report,
conservative typed NAL registration and exact/semantic readback, then commit
only this study's source/tests/docs. Preserve all unrelated dirty files.

Method references: [Neighbourhood Distillation](https://arxiv.org/abs/2010.01189)
and [SVD-LLM](https://arxiv.org/abs/2403.07378). This is an exploratory Llama
adaptation; it does not reproduce either paper's complete experimental setup.
