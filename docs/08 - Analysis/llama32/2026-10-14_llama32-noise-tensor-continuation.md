# Llama 3B tensor continuation: proxy quality recovered, speed target unmet

Date: 2026-10-14. Status: completed EXPLORATORY / UNDERPOWERED.
Hypothesis: `llama32-3b-noise-tensor-continuation-v1`.
Classification: `proxy_quality_recovered_deployment_speed_unproven`.

The selected noisy tensor student passed **20/20 complete final-test requests**,
versus **19/20 for original FP16 and 19/20 for native Q4+lookup**. It passes the
prespecified >=95% synthetic quality gate without fewer passes than either
control. Both clean and noisy students passed 8/8 development requests, so
**Gaussian noise did not demonstrate a task-accuracy advantage**. The assembled
FP16 weight footprint is 35.64% below the original FP16 model, but the prototype
remains substantially slower than the deployed native Q4 baseline. Production
application accuracy and the user's 2x deployment-speed goal remain unproven.

Aggregate: `data/experiments/llama_neighbourhood_continuation/20261014_registered_v1/campaign_results.json`.
SHA-256: `f4215b2c3a453320ec91835f7110966b1db45668699fa940e2d019fecc6b9abd`.
The [prospective plan](../07%20-%20Status%20Reports/2026-10-14_llama32-noise-tensor-continuation-plan.md)
and frozen producer snapshots define the comparison. The failed
[parent experiment](2026-10-14_llama32-neighbourhood-tensorization.md) remains
unchanged; this follow-up does not reclassify it.

| Model | Complete development requests | Complete final-test requests |
| --- | ---: | ---: |
| Parent locally distilled tensor student | 0/8, reused | Not executed |
| Clean continuation + global recovery | 8/8 | Not selected |
| Noise continuation + global recovery | 8/8 | 20/20 |
| Original FP16 | 8/8, reused control | 19/20, fresh |
| Native Q4 + prompt lookup | 8/8, reused control | 19/20, fresh |

A request passes only when every fragment has its current owner, approved
budget and deadline, with correct attribution, no invented or superseded fact,
no missing/extra fragment, and <=128 tokenizer tokens per stored summary.
Labels on their own heading lines are accepted under the inherited fixed
protocol; literal one-line formatting is reported separately in reviews.
Judging is explicit assistant source review plus token/cardinality checks;
there is **no independent judge model**. This is a narrow synthetic final-decision
proxy, not a representative production summarization evaluation. The selected
student reproduced all final-test references and stopped at EOS; its longest
stored summary was 33 tokens. Reference equality is an observed property, not
a required exact-output criterion.

The FP16 control failed `heldout_raw_23018_8_6220`: F01 received F06's $1,800
and November 11 instead of its correct $8,700 and November 12. Q4 failed
`heldout_raw_23010_8_6220`: F01 received F05's $4,800 and November 16 instead
of its correct $2,300 and November 13. Both are attribution errors between
fragments sharing an owner. These misses also show why agreement with the
original model cannot replace source-grounded correctness.

Both arms use the full 28-layer Llama 3.2 3B Instruct decoder, with rank-1024
factors for each gate/up/down MLP projection. Attention, embeddings, norms and
normal residual connections retain their original weights. There is no dense
exact correction and no new faster-attention implementation in this study.
The starting factors came from original-weight, activation-scaled SVD and the
parent's 96 local updates/group. They were not freshly randomized. Independent
deserialization confirmed that all 14 clean initial states exactly equal the
parent's selected FP16 states. Adam state was unavailable: continuation resets
optimizers and promotes saved FP16 factors to FP32 masters; it is weight-only
continuation rather than exact optimizer resume.

The noise package applies 1% factor-RMS Gaussian weight noise once and 5%
channel-RMS Gaussian input-activation noise on every second local update.
All 168 saved initial factor perturbations were checked: relative RMS ranged
from 0.0099919 to 0.0100138. Each perturbed input goes through both frozen
original teacher blocks and student blocks, with a freshly recomputed teacher
target. The 1,792 noisy local updates never use stale original-input targets.
Noise/data-order streams are separate and the arms receive identical example
schedules. This combined treatment does not identify weight noise and input
noise independently. Inference adds no noise.

Each arm completed 256 additional local updates for each of 14 two-layer groups,
then 32 global updates for each active group in order 0–13: 4,032 updates per
arm, **8,064 additional optimizer updates** in total. Local AdamW LR is 2e-5,
zero weight decay, clip norm 1; the loss equally weights all-position and
response-position per-token relative MSE. Development selection uses actual
FP16 factors at 0/64/128/192/256; all groups selected step 256. Global AdamW
LR is 1e-5 and its loss is 0.5 teacher-distribution KL at temperature 1 plus
0.5 reference-token CE. Non-reentrant checkpoints let the full assembled decoder
participate while only one factor group is trainable. This is block-coordinate
optimization of a global loss, not simultaneous optimization of every student
parameter as in a full Puzzle training run. Global checkpoints are the fixed
final step, not best-development snapshots.

| Arm | KL before global / after global | Reference CE before / after | Local + global group time |
| --- | ---: | ---: | ---: |
| Clean | 0.035509 / 0.047970 | 0.113988 / 0.024378 | 1,259.41 s |
| Noise | 0.046103 / 0.034561 | 0.116516 / 0.029367 | 1,368.24 s |

Both arms pass 8/8, so lower final development KL selects the noisy arm for
final testing. The clean arm's KL worsened while CE improved: the mixed loss
trades teacher fidelity against the supervised reference, and KL is not the
quality gate. The old-to-new comparison changes local training budget,
response weighting and assembled recovery together. We did not generate a
local-only continued-model ablation, so we cannot attribute the quality
recovery solely to the global stage. The noise comparison is one paired seed
and cannot establish a general robustness benefit.

The data comprise 64 training requests, eight development requests with known
parent outcomes, and 20 final-test requests unused for training/selection.
The final test was executed only after both development reviews and candidate
selection. Inputs cover roughly 500/2,000/6,300-token raw prompts and 1,000-token
hierarchy prompts, with up to eight fragments. The same 128-token storage rule,
2,048-token fragment ceiling and 768/512 generation caps apply. Short-text
bypass, broad abstractive summaries and application retrieval/hierarchy accuracy
are not tested. Cached teacher states are training resources; runtime KV caches
are fresh per request, with no cross-request prefix reuse.

| Runtime | Assembled weight bytes, decimal GB | Development fixed-32 tokens/s |
| --- | ---: | ---: |
| Clean FP16 factor prototype | 4.135 | 12.086 |
| Noisy FP16 factor prototype | 4.135 | 11.190 |
| Native Q4 + lookup-8 | 2.012 | 42.926 |

The original FP16 footprint is 6.425 GB. Prototype weight bytes count actual
assembled parameters, including tied embeddings only once; they are not a
standalone exported package size. The loader still obtains frozen base weights
from the original HF checkpoint. Factorization reduces FP16 parameter bytes by
35.64%, but its FP16 representation remains larger than native Q4. Both factor
arms have identical architecture and work shapes; their timing difference does
not establish a computation reduction caused by noise.

Fixed timing uses identical tokenized development prompts, 32 forced token
steps including special tokens, four input families and three repeats. It
excludes model load, tokenization and detokenization. PyTorch uses FP16,
FlashAttention2 and bounded 512-token prefill chunks, without CUDA graphs or
prompt lookup. The native control uses quantized weights, CUDA graphs and
within-request lookup drafting. This is a deployment-system diagnostic with a
backend/precision mismatch, not a causal factorization-only speed comparison
or a five-repeat useful-summary benchmark. On the paired final-test fixed-32
PyTorch diagnostic, noisy factors measured 11.912 tokens/s and original FP16
11.863: only 0.4% difference, not a decisive speed improvement. Mean natural
final-test request times were 6.542 s for the student, 5.582 s for original FP16
and 1.034 s for Q4; mean generated lengths were 89.5/82.7/87.2 tokens, so those
means are not length-matched throughput claims.

Execution used only physical GPU 0, RTX 3060 12 GB, serially, with two CPU
threads; GPUs 1/2 remained reserved. The runtime is Python 3.11, Torch 2.4.1
CUDA 12.1, Transformers 4.53.3 and FlashAttention2 2.6.3. CPU/contract tests
passed (nine tests). The full-size local-noise lifecycle and longest-input
full-decoder pilots passed under separate non-quality roots. All 28 local
and 28 global groups completed, with finite nonzero active-factor gradients,
frozen original weights and exact FP16 checkpoint reloads. Peak allocated
training memory was 7,024,784,896 bytes. No study run failed or was retried.
The aggregate has 192 direct training/evaluation/timing cells, not 192 independent
replicates; systems pilots and reused parent controls are separate. Group times
include fitting/validation/checkpoint work and exclude initial full-model load.

The [research review](2026-10-14_embedded-neighbourhood-distillation-research.md)
identifies [Neighbourhood Distillation](https://arxiv.org/pdf/2010.01189) and
[Puzzle](https://proceedings.mlr.press/v267/bercovich25a.html) as motivation for
local fitting plus assembled recovery. The [embedded quantization follow-up](2026-10-14_embedded-quantized-neighbourhood-followups.md)
connects EfficientQAT, MobileQuant and T-MAC to deployable low-bit operators.
A next measured study should preserve these quality gates while testing
hardware-supported widths/ranks and quantization. Native factor kernels,
factor quantization, a combined two-model deployment, phone performance and a
five-repeat quality-matched speed campaign remain unexecuted. Planning arithmetic
about rank 512, FFN width 1,920 or shared immutable base weights is not a measured
compression/speed result for those configurations.

Reproduction uses the cached original HF snapshot and immutable parent cache.
Use a fresh root in place of the producing root below; do not overwrite completed
evidence. The coordinator requires the passed pilot roots and runs both arms
serially:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
LD_LIBRARY_PATH='' data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.campaign_llama_noise_tensor_continuation \
  --root data/experiments/llama_neighbourhood_continuation \
  --prepared data/experiments/llama_neighbourhood_continuation/20261014_prepared_v1 \
  --local-pilot data/experiments/llama_neighbourhood_continuation/20261014_local_noise_pilot_v1 \
  --global-pilot data/experiments/llama_neighbourhood_continuation/20261014_global_memory_pilot_v1 \
  --output data/experiments/llama_neighbourhood_continuation/20261014_campaign_v2
```

Training/evaluation source is `experiments/structure_net/llama_noise_tensor_continuation.py`;
conditional final testing uses `campaign_llama_noise_tensor_heldout.py`.
The finalizer audits source snapshots, lineage, schedules, checkpoints, complete
outputs, native configuration and all 76 source reviews (16 development plus 60
final-test arm/request reviews). Registration conservatively keeps
`confirmed: false` and verifies exact and semantic readback. The source-grounded
quality gate passes on this proxy; the full accuracy-and-deployment-speed objective
does not pass.
