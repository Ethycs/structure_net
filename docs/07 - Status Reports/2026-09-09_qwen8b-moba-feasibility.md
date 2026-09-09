# Qwen3-8B prefix: MoBA feasibility experiment

Date: 2026-09-09. Profile: EXPLORATORY_FEASIBILITY.
Hypothesis: `qwen3-8b-prefix6-moba-feasibility-v1`.

The user authorized representative experiments to establish whether the MoBA
paper can apply to Qwen 8B without requiring their exact application or model.
Their described workload runs five full transformer blocks, then extracts
attention readouts from block six without its MLP. We test that public-model
prefix in FP16, with FP32 routing/softmax accumulation and pooled readouts.
No gateway requests, summarization service, application edits or full-LLM
training are part of this experiment.

## Inputs and implementation

Use public `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`. Its first six layers and embedding
require 67 tensors / 1,780,008,448 parameters (3.560 GB FP16), all available in
`model-00001-of-00005.safetensors`. Download only that shard, tokenizer and
configuration, and hash them. Omit final LM head and layers 7–36. The sixth MLP
may be loaded for strict checkpoint identity but is never executed.

Use the official MoBA source at commit
`b5d58363311d3ca946f1ec444182727c15e338b5`, FlashAttention 2.6.3 and isolated
PyTorch 2.4.1+cu121. The first source build against shared PyTorch 2.5 was
stopped before any evidence run; an official ABI-matched wheel replaces it.
The shared environment remains unchanged. Full attention uses optimized
FlashAttention, preserving Qwen's grouped-query semantics. Charge MoBA for KV
head expansion required by the current upstream implementation, routing,
reordering and output combining. Apply Qwen's actual Q/K normalization, RoPE,
projection weights and residual/MLP graph in the first five blocks.

Use fixed natural-language windows from the existing BabyLM Wikipedia and
Switchboard source files, retokenized with Qwen's tokenizer. These are generic
language inputs, not the user's actual summary distribution. Hash sources,
tokens and exact window schedules. Within each source's token stream, use the
first 60% for optional adaptation, the next 20% for configuration selection,
and the final 20% for held-out feasibility measurement; windows may not cross
partitions. Use seeds 7,17,29,41,53 for independent window schedules.

## Measurements and meaning

Pool the sixth layer's attention-weighted value vectors and projected attention
output into consecutive 128-token segments, accumulating in FP32. Compare with
the unchanged full-attention prefix. Report segment cosine similarity, relative
L2 error, adjacent-segment change-score correlation, and agreement of the top
quarter of changes. These explicitly defined readout-preservation proxies do
not equal hierarchy split accuracy or retrieval accuracy. The application's
95% final accuracy cannot be established from these unlabeled inputs.

The operational candidate screen is mean segment cosine >=.95, change-score
Pearson correlation >=.95, top-quarter change F1 >=.90, and measured prefix
speedup >=1.05 against optimized full attention. These are feasibility gates,
not a redefinition of application accuracy. Report every endpoint and every
failed setting. A positive result motivates application evaluation; a negative
result rejects the tested setting only.

Report both isolated-attention and six-layer-prefix timings, with median and
sample spread from alternating paired CUDA-event measurements. Include one
warmup followed by five paired samples; repeat count may be one for expensive
prefix runs. Disable TF32 and record GPU, dtype and source identity. Do not time
the measurement/readout comparison itself or disk I/O. Both timed forward paths
must include the same pooling/projection output contract. Peak memory is also
reported. CUDA graph timing, if added, is supplementary and explicitly labeled.

## Decision sequence

1. Verify official sparse forward/backward parity against its dense reference,
   partial blocks, and full-budget equality with optimized full attention.
   Observed pilot FP16 maximum output error was 9.77e-4; maximum gradient error
   was 3.91e-3 on four fixed shapes, within .02 absolute/relative tolerance.
2. Verify the manual six-layer prefix against the unmodified Hugging Face
   decoder graph on a small deterministic input, omitting only the sixth MLP.
   Run a real prefix CUDA memory and readout lifecycle pilot at length 512.
3. Screen zero-shot MoBA on validation windows at lengths 512, 2048, 8192 and
   16384. Fixed settings: block128/top2, block128/top4, block128/top8,
   block256/top4. Compare applying MoBA to all six attention layers, the first
   five with full final attention, and the final layer only. If selected blocks
   cover the whole context, transparently use full attention and label the case
   full coverage, with no sparse-acceleration claim.
4. Freeze the best passing configuration by prefix speed on the 8192-token
   validation window (ties: higher change correlation, then listed order).
   Evaluate that setting on five fresh windows, paired by input, reporting
   readout gates and measured timing. Also report the length scaling curve.
5. If no zero-shot setting passes but some offers >=1.05 prefix speedup, try
   attention-only adaptation on the fastest setting and the fastest setting
   with cosine >=.90. Use rank8/alpha16 adapters on Q/K/V/output projections of
   the six attention modules only, frozen original weights, FP32 adapter and
   optimizer state, AdamW LR1e-3, betas(.9,.95), weight decay.01, clipping1,
   128 updates of one 2048-token training window. Distill full-prefix head and
   projected-output representations, each normalized MSE with equal weights.
   Teacher uses the same frozen backbone with adapters disabled and full
   attention. Check validation at 0/32/64/96/128; select first passing checkpoint
   using the same gates and no final-test peeking. Preserve all checkpoints.
   If no setting offers a speedup, diagnose the bottleneck before spending
   adaptation compute; no quality change can rescue a slower kernel.

Training, if reached, must hash-check all non-adapter weights before/after and
verify no forbidden parameter acquires a gradient. It does not retrain MLPs.
Any extension requires a dated amendment and fresh artifact root.

## Evidence discipline

Entry point: `experiments/structure_net/qwen_moba_prefix.py` and a dedicated
campaign driver. Root: `data/experiments/qwen_moba_prefix/20260909_<stage>_v1/`.
Use one worker on physical GPU 0, preserving other workloads. Snapshot and hash
all outcome-relevant source/config/checkpoint/data dependencies before each
stage. Save strict JSON, exact inputs, direct metrics, timings, gates, failure
records and completion counts. Completed roots are append-only.

No claim of broad Qwen benchmark accuracy, 95% retrieval accuracy, decode
acceleration, whole-8B-model speedup, or exact application replication follows
from this prefix feasibility study.

Sources: [MoBA paper](https://arxiv.org/pdf/2502.13189),
[official implementation](https://github.com/MoonshotAI/MoBA),
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

## Pre-execution corpus-size amendment, 2026-09-09

The systems pilot stopped before loading the model because the Switchboard
source contains only 155,042 Qwen tokens. Its final 20% cannot supply five
disjoint 16,384-token windows. No primary or held-out result was inspected.
Replace Switchboard with the existing larger BNC spoken corpus; retain
Wikipedia, all split fractions, disjoint schedules, settings and gates. Record
this failed systems attempt under `20260909_pilot_v1/` and use new
`inputs_v2/` and `20260909_pilot_v2/` roots. This is a data-availability
correction, not selection using quality outcomes.

## Pre-screen context-range amendment, 2026-09-09

The completed systems pilot verified the real prefix graph and frozen weights;
the 512-token case was slower. No primary grid or final readouts have been
examined. Include 32,768 tokens to cover the paper's long-context regime within
Qwen3-8B's native configured maximum. Both retained corpora have enough tokens
for five disjoint final windows at this length. Repeat the systems memory pilot
at 512 and 32,768 before screening, under `20260909_pilot_v3/`.

Selection now chooses the shortest of 512/2048/8192/16384/32768 where a setting
passes every feasibility gate on both corpus validation windows. Among passing
settings at that length, choose the greatest mean prefix speedup, then greatest
mean change correlation, then listed order. Freeze both setting and length for
the five-seed final measurement on both corpora. The prior 8192-only selection
rule is superseded before primary outcome inspection. This range tests where
the method first becomes useful; it does not claim short-context acceleration.

## Resource amendment before primary screen, 2026-09-09

The 32K systems pilot failed in the full baseline's MLP allocation after a
concurrent job in another repository occupied 4.96 GiB and saturated GPU 0.
Its source, completed 512-token cell and failure are preserved under
`20260909_pilot_v3/`. Do not treat this as a MoBA quality failure or use its
contended timings as primary speed evidence.

Bound the unchanged tokenwise MLP computation to 2048-token chunks in both full
and MoBA paths, freeing unused attention temporaries before each MLP. Verify
equivalence against the unchanged graph. For readout collection while the GPU
is shared, the official MoBA operator may process four query heads (one complete
GQA group) at a time, with a 45% per-process CUDA allocator limit. Verify head
ordering and output parity; this changes scheduling, not attention selection.
Save explicit head-chunk and memory-limit configuration in every producer.

`--stage readouts` collects the fixed validation grid with timing and joint
gates left null; it cannot select a candidate or claim speed. Run the timed
screen with all 32 heads together only once the GPU is available, using a new
root and paired full/MoBA timings. Preserve the readout-only evidence as an
auxiliary systems-constrained stage. CPU tests cover token/head chunking; repeat
the real-model memory pilot before this collection. Never terminate unrelated
workloads to obtain a benchmark slot.
