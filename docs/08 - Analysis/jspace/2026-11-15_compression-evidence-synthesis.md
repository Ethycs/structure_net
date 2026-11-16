# Compression evidence: what transfers from local fits to a usable model

Evidence snapshot: **2026-11-15, 22:30 PST / 2026-11-16, 06:30 UTC**.
This synthesis connects completed findings and explicitly dated running work to
the [design](../../01%20-%20Design/jspace/residual-compression-and-recovery.md)
and [implementation](../../02%20-%20Implementation/jspace-compression-and-recovery.md).
It supplements the frozen reports; it does not amend their protocols or results.

The research has demonstrated useful compression of a real Qwen block, including
measured local acceleration and fresh task-quality checks. It has also shown
that independently successful blocks can fail badly when assembled. Progressive
recovery now preserves much more short multiple-choice performance through a
15-block suffix, but that result has no new generation or independent quality
qualification. A usable fully compressed model remains an open result.

## Evidence that changed the design

| Evidence scope | Observed result | What it establishes and changes |
| --- | --- | --- |
| Finite country example | An exact codec reduces 128 nine-layer label trajectories from 298 to 58 bytes; exception-preserving swap trajectories take 62 bytes. The selected continuous quadratic surrogate preserves all nine labels on only 96/128 natural examples. | Exact elimination and decoding work for the constructed finite representation. The label codec does not reconstruct Qwen states or remove transformer layers; an exact surrogate certificate does not certify neural fidelity. [Report](2026-11-13_jspace-country-stack.md), [aggregate](../../../data/experiments/jspace_country_stack/20261113_v1/campaign_results.json). |
| Useful isolated final block in Qwen2.5-3B | Fast compact: **391/512**, native: **393/512**, on a fresh set. The whole final block is **1.297× faster in prefill and 1.275× in cached decode** against the native block, with both graph captured. | A bounded positive result combining functional compression, recovery and execution optimization. Whole-model graph throughput remains about 41 tokens/s with negligible improvement; only one of 36 blocks changed. [Report](2026-11-13_jspace-compact-performance.md), [aggregate](../../../data/experiments/jspace_fast_recovery/20261113_v2/campaign_results.json). |
| Earlier block, different architecture | Rank-1,024 block 0 with full residual carry/native-width heads improves **27→46/64** after basis enrichment, then **46→48/64** after isolated recovery; matched FP32 native is **50/64**. | Geometry selection and learned realization are separate interventions. The larger 27→46 gain belongs to basis construction; it is not a distillation gain. [Evidence](2026-11-14_qwen-single-layer-recovery-result.md). |
| Independent fits assembled | Blocks 2–35: **34 verified**, **32 isolated passes**, misses at 12/15. Complete all-compact model: **27/64**; hybrid retaining native 12/15: **24/64**; accepted two-block control: **49/64**; native: **50/64**. | Local task preservation does not compose automatically, and restoring the isolated misses did not repair the stack. Measure the complete raw-token path and fit against the actual downstream model. [Workers](../../../data/experiments/jspace_parallel_single_layer/20261114_v1/result.json), [assembly](../../../data/experiments/jspace_recovered_assembly/20261114_v1/paired_result.json). |
| Progressive interface and joint recovery | The present 15-block suffix selects update 288 at **189/256**, versus original native **195/256**, with exact fresh replay. Its warm initializer scores **177/256**. | Recovery can improve the tested assembled suffix after adding an entrance block. Passing reused DEV permits exploratory continuation; it does not establish general language recovery. [Training](../../../data/experiments/jspace_3b_broad_recovery/20261115_v2/progressive15/train/result.json), [verification](../../../data/experiments/jspace_3b_broad_recovery/20261115_v2/progressive15/verify/result.json). |
| Independent all-36 joint run, interim | Initializer **103/256**; update 72 **109/256**, against native **195/256**. Training was still in progress at this snapshot. | All blocks can participate in real recovery updates; useful complete-model quality is not yet demonstrated. This is a separate experiment, not a matched comparison proving a causal advantage of progressive training. [Initial](../../../data/experiments/jspace_3b_broad_recovery/20261115_v2/all36_joint/train/update_000_development.json), [update 72](../../../data/experiments/jspace_3b_broad_recovery/20261115_v2/all36_joint/train/update_072_development.json). |

These rows are not a single monotonic benchmark series. The country labels,
continuous readouts and Qwen answers are different measurements. The historical
single-block/independent campaign used full-width residual carry, native
128-dimensional heads and FP32 task checks; the current suffix uses affine
residual coordinates, 64-dimensional heads, polynomial MLP substitutions and
BF16 execution. The historical 64-case native FP32 score of 50 and BF16 score
of 48 are not interchangeable with the newer 256-case native score of 195.

The final-block speed result also has two distinct comparators: **1.999×**
prefill was relative to the older compact executor; **1.297×** was relative to
the original native block. Neither ratio is a whole-model speedup. Its fresh
512-case set belongs to that historical study, not to the still-sealed final
confirmation set of the present broad-recovery campaign.

## Current results, with the denominator and lineage attached

The broad recovery uses **4,608 TRAIN and 256 reused DEV** examples from
ARC-Challenge, BoolQ, HellaSwag and WinoGrande. Each DEV family has 64 cases.
These are short multiple-choice samples, not complete benchmark suites.
Counts below were recomputed from saved answer/prediction records; the 15-block
training and fresh-verification records and metrics agree exactly.

| Model / checkpoint | ARC | BoolQ | HellaSwag | WinoGrande | Total | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original native reference | 55 | 54 | 47 | 39 | 195/256 | 76.171875% |
| 15-block suffix, update 288 | 51 | 52 | 48 | 38 | 189/256 | 73.828125% |
| All-36 initializer | 21 | 31 | 17 | 34 | 103/256 | 40.234375% |
| All-36, update 72 | 18 | 40 | 14 | 37 | 109/256 | 42.578125% |

The 15-block result is **2.34375 percentage points below native**. It meets
the fixed descriptive DEV limits: at least 183/256 overall and at most six
net lost answers per family relative to original native. Its selected
checkpoint is `progressive15/train/update_288.pt`, SHA-256
`22adc8b9870b294b8a3405fb7b3be511ea2f0bed6bdef0ae7fb74dc479c2b1be`.
Strict state loading, unchanged fixed state and exact output replay passed.
The receipt still records `promoted:false`, `confirmed:false`,
`speed_measured:false` and `full36:false`.

The all-36 aggregate gain of six answers includes ARC and HellaSwag regressions
of three each, plus BoolQ/WinoGrande gains of nine/three. Its saved loss drops
from 20.5344 to 14.3181, yet the task score remains far below the DEV gate.
This is why loss, total accuracy and family outcomes must remain separate.
At the snapshot, the latest recorded training update was **117/288**
(`2026-11-16T06:28:54.820964+00:00`); **72 was the latest scored checkpoint**.
An update receipt is not a newer accuracy result or a completed selection.

The primary progressive continuation is
`/data/structure_net_experiments/jspace_3b_progressive36/20261115_v1`.
Its [events](/data/structure_net_experiments/jspace_3b_progressive36/20261115_v1/events.jsonl)
show the 16-block stage completing boundary, capture, fit and parent-feature
capture, then starting its discarded pilot at **06:29:06 UTC**. There was no
selected or verified 16-block recovery result at this snapshot. The independent
all-36 run remains a separate initializer, training path and outcome.

The difference between the 15-block and all-36 scores is descriptively useful,
but depth, inherited learned weights, calibration history, parent-feature
supervision, microbatching and optimizer placement differ. Their separation
cannot isolate which intervention causes the performance gap. The
[all-36 protocol](../../07%20-%20Status%20Reports/jspace/2026-11-15_qwen3b-all36-joint.md)
explicitly records these differences.

Both paths reuse pretrained computation: the all-36 initializer projects native
parameters into fitted geometry, while progressive expansion also inherits its
recovered suffix. Their small recovery budgets are adaptation budgets, not
training a language model from scratch. No matched random-initialization or
source-prior ablation establishes a numerical sample/compute-efficiency gain.

Phi is **paused at the user's request** after native capture. Its **58/64**
DEV score is the GPTQ source decoded into BF16, a different parent and task
sample; it is not a compressed Phi result or an original full-precision
baseline. No Phi fitting or recovery was launched. [Native/pause record](2026-11-15_phi4-pilot-launch.md).

## What the failure evidence teaches

**Task selection and generation must be checked separately.** The 14-block
parent also scored 189/256 on DEV, but its subsequently opened 128-case audit
scored **93/128 versus native 103/128**, a **7.8125-point** loss that failed
the five-point screen. BoolQ was 22/32 versus 28/32. The 15-block continuation
explicitly admits this failed parent as exploratory; its newer DEV result
does not erase that failure or constitute a new audit.

Both the 14-block seed and recovered parent showed repetitive-label or
incoherent collapse on **6/6** frozen generation canaries. Native had **0/6**
collapse flags, although its strict fulfillment was only **1/6** because the
64-token budget and formatting/detail requirements also exposed native
failures. This was qualitative review of six prompts, not a population measure
or an independent judge. **Generation for the selected 15-block checkpoint
has not been measured.** [Assessment](../../../data/experiments/jspace_3b_broad_assess/20261115_v1/result.json),
[semantic review](../../../data/experiments/jspace_3b_broad_assess/20261115_v1/semantic_review_v1/result.json).

**Full vocabulary does not mean full response coverage.** The broad native-content
teacher supervises three interior content positions and one answer position
per example: 18,432 positions among 622,447 TRAIN input tokens, or 2.96%.
There are no generated continuations in those targets. Earlier parent-legacy
targets use a different positional recipe. Their KL terms are full vocabulary
at the selected positions; they do not train every answer token or ordinary
generation. Likewise, a full-vocabulary random VJP probe explores local output
sensitivity only on its sampled contexts/positions; it is not a generation
quality test. [Coverage audit](2026-11-15_qwen3b-broad-recovery.md),
[legacy position audit](2026-11-15_qwen3b-cached-position-loss-audit.md).

**A good local fit is useful evidence with a bounded scope.** The single-block
result and assembly failure together support downstream-aware calibration and
explicit interface tests. Aggregate feature error alone did not distinguish
the independently failing blocks from successful neighbors; feature loss
could fall while accuracy declined. Projection controls diagnose representable
feature error but are not task-accuracy ceilings. [Independent failure analysis](2026-11-14_jspace-independent-layer-failures.md).

**Implementation fidelity and functional recovery answer different questions.**
Fresh replay, correctly constructed rotary state, exact inherited bridges and
protected initializers prevent deployment or selection errors. They cannot
certify general ability. The early full-compact branch's omitted value-mean
correction was an implementation defect, while its corrected approximation
still required quality testing. Preserve both outcomes and distinguish a failed
execution control from a scientifically measured quality miss. [Corrected full-compact report](2026-11-13_jspace-full-compact.md).

**Efficiency requires a matched deployment measurement.** The failed historical
all-compact assembly saved 42.48% of matched FP32 model tensor bytes, including
geometry, but was only 27/64; this is not a usable compression result or a
comparison against BF16 native storage. The successful isolated final-block
study measured local speed and bounded fresh quality together. Neither result
establishes a current 2× whole-model speedup or 40% reduction with preserved
quality. Training speed, optimizer spill, parameter counts and theoretical
arithmetic savings cannot substitute for exported tensor bytes and isolated
end-to-end prefill/decode measurements.

## Next evidence needed to change the conclusion

The existing progressive run can establish further **structural coverage and
DEV continuation** under its frozen sparse objective. Complete-model recovery
requires all 36 selected blocks, exact raw-token replay, a separately frozen
independent quality evaluation and generation checks on the exported model.
The exposed 128-case audit cannot become a fresh test again; repeated DEV
selection also increases selection pressure.

The new [complete-response recovery implementation](../../02%20-%20Implementation/distillation-recovery.md)
addresses the coverage limitation with response masks, observed EOS handling,
chunked full-vocabulary loss and authenticated rollout targets. Its current
evidence is **CPU systems qualification**, including tiny compiled-Qwen
training/reload and a persistent two-update synthetic run. It has not trained
the actual 3B model or demonstrated recovery of its generation. Corpus
admission, actual model/resource fit, generation-aware selection and measured
quality remain required. [Qualification report](2026-11-15_distillation-standard-implementation.md).

This synthesis used local reports and small JSON/JSONL receipts only. It did
not load model weights, launch inference/training, reopen sealed evaluations,
modify frozen artifacts or rehash large banks. Checkpoint identities above
are the recorded producer digests; only saved prediction counts and the
15-block report/replay equality were independently recomputed here.
