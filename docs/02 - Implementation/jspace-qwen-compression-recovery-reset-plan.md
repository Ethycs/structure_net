# Qwen J-space compression: recovery and evaluation reset plan

Date: 2026-11-17. Status: **planning only; no new experiments authorized by this document**.

## 1. Decision and execution state

Stop adding architectural changes and small bespoke experiments until we establish a credible recovery and evaluation procedure. Preserve the J-space / polynomial / compact-plus-residual research direction. The immediate question is whether adequately funded recovery preserves useful behavior at a fixed compression level—not whether a lightly recovered model already passes every task.

On the user's stop-and-plan instruction:

- Cancelled the two new MMLU-Redux implementation agents. No new MMLU inference was launched.
- Stopped the deferred quadratic-attention controller while it was still `waiting_parent`, before it acquired a GPU reservation. Its earlier implementation and test receipts remain preserved.
- Preserved the existing progressive campaign, three-block checkpoint evaluations, and ABLATION32 search rather than terminate training or grading mid-phase. **These existing campaigns remain active; this is not a claim that every GPU job was stopped.** Their subsequent results belong to their original protocols.
- The completed CPU benchmark-length preflight remains available. Downloading and measuring prompts is not scoring the benchmark or establishing contamination freedom.

This plan supersedes the *proposed next-work ordering*, not frozen historical protocols. Implementation starts only after plan review. No extra controller, GPU reservation, or recovery arm is launched as part of writing it.

## 2. Objective and invariants

**Primary objective:** retain as much original-teacher capability as possible while compressing all 36 blocks, with the user's allowance of up to **5 percentage points** of task-accuracy loss. Report usefulness relative to the teacher separately from absolute application usefulness.

**Deployment aspirations:** approximately 38% smaller and 1.4× faster, as in the supplied small-model example. These remain targets, not demonstrated Qwen results. Count the entire deployed model, not only replaced blocks.

Keep these distinctions explicit:

- Teacher: **Qwen/Qwen2.5-3B-Instruct**, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16. Do not substitute base-model, Qwen3, or larger-model reference scores.
- Current compression uses functional task-Jacobian/VJP-derived bases. Mean activations supply affine anchors; they are **not** the criterion selecting the retained functional subspace.
- Current reduced heads retain softmax. The existing polynomial MLP and the proposed quadratic attention are separate interventions.
- ABLATION32 learns discarded directions using finite-removal output KL. It is not the complement of an average activation.
- Preserve uncompressed/residual operations explicitly. Inventory what is retained, approximated and discarded; do not claim losslessness from a fitted basis or from successful distillation.
- Gröbner reduction is principled algebra given the specified relations. Approximately fitted relations still require held-out functional validation; their existence alone does not prove equivalent language-model behavior.

## 3. What our experiments actually establish

### Fixed checkpoints—not interchangeable descriptions of “the model”

| Artifact | Observation | Supported interpretation |
| --- | --- | --- |
| Progressive eight-block checkpoint, update 46 | MC 195/256, equal to teacher 195/256; task-success judgments 1/16 vs teacher 4/16; reliable generations 4/16 vs 8/16; canaries 0/6 vs 2/6 | Retains this MC screen; generation recovery remains incomplete on the small screen |
| Progressive nine-block checkpoint | MC 196/256 vs teacher 195/256 | No material measured MC degradation; one extra answer is not proof of improvement |
| Separate three-block full-corpus recovery | 4,441 TRAIN examples, 105,089 response/EOS tokens and 690,623 input tokens, one pass | A larger recovery pilot, still not a large-scale recovery study |
| Three-block updates 0 → 71 → 134 | Useful completed tasks 0 → 2 → 3 out of 16; reliable generations 4 → 7 → 9 | Recovery improves generated behavior on exposed DEV; depth is unchanged |
| Three-block update 134 paired teacher | Useful completed tasks 3/16 for both; reliable generations student 9/16 vs teacher 8/16 | Encouraging relative retention, not general-purpose equivalence |
| Historical all-36 quadratic attention | 220/512 vs compressed softmax 211/512, native 396/512; paired quadratic-minus-softmax bootstrap interval −3.125 to +6.641 pp | Large overall recovery failure did not isolate attention as its cause; neither attention equivalence nor improvement was established |

Eight-block checkpoint:
`/data/structure_net_experiments/jspace_3b_full_progressive/20261116_v1/blocks_08/recovery/run/checkpoints/update_0046.pt`

SHA256: `9d2e43b25aa07bc34f549cc91b6e95258bf08721733990ff10c8c51704de6958`.

Three-block results:
`/data/structure_net_experiments/jspace_3b_usable_recovery/20261116_v1/stage_a/`.

Historical attention reanalysis:
`/data/structure_net_experiments/jspace_recovery_robustness/20261116_v1/result.json`.

The progressive chain still used approximately 25k response/EOS tokens per recovery stage. Its deeper checkpoints did **not** automatically receive the separate three-block full-corpus recovery recipe. Earlier layer fitting and previous recovery are additional ancestry; 25k is the per-stage budget, not total lifetime supervision.

### Measurement and execution limitations

1. Six canaries and 16 generated cases are diagnostics, not a five-point population accuracy test. A one-case canary change is 16.7 percentage points on that screen.
2. Repeated paired semantic judgments changed the same teacher's task-success count from 3 to 5/16, while useful completions remained 3/16. Judge variability must not masquerade as model movement.
3. MC answer ranking and teacher-forced NLL do not measure long-form generation. Neither can substitute for it; neither should be discarded merely because the teacher has weak absolute generation scores.
4. All current inspected compact exports use **416 total positions** with an error policy beyond the fitted position anchors. The 128-token generation cap is another, separate constraint. Raising a configuration limit alone does not extend the model.
5. The new zero-shot prompt preflight found 2,973/3,000 MMLU-Redux v1 prompts fit 416 positions with one output token; all 1,319 GSM8K prompts fit with 128 output tokens; 149/164 HumanEval prompts fit with 128 output tokens. These are input-budget checks, not evidence that 128 tokens suffice to solve the tasks.
6. Existing benchmark DEV is exposed. The prior final512 data-access incident remains recorded; that set cannot be described as untouched. A fresh download is not automatically independent of recovery training.
7. Historical single-block kernel gains and cache-free, host-offloaded generation timings do not establish a current whole-model speedup. In `attach_prefix(mode="cpu")`, native weights reside on the host but computation streams through the GPU; it is not CPU-only prefix computation.

Preflight evidence:
`/data/structure_net_experiments/jspace_3b_benchmark_coverage/20261117_v1/report.json`.

## 4. Research implications—not promises transferred from other models

| Research | Relevant evidence | Consequence for our design |
| --- | --- | --- |
| [SliceGPT, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/316648eb8b4ffb6010f531b07848c300-Paper-Conference.pdf) | Smaller dense matrices can produce real runtime gains; recovery depends on calibration data, with instruction data benefiting Phi-2 more than WikiText-2 | Preserve compiled dense execution and normalization/bridge correctness; validate the data distribution and measure complete-model runtime |
| [Lillama](https://arxiv.org/html/2412.16719v2) | Identifies teacher-only versus student-input activation mismatch; combines both. Uses 13M calibration tokens and evaluates nine downstream tasks | Layerwise success is not enough. Fit/recover on the actual compressed prefix and test a materially larger diverse corpus |
| [Minitron, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf) | Compression rankings can change after retraining; iterative prune/distill and logit KD work well in its experiments. Even its lightweight ranking studies use about 1.8B tokens | Do not judge representational capacity from a minimal recovery budget. Logit KD is a sound baseline, not proof that our current budget is adequate |
| [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) | Teacher feedback on student-generated histories addresses autoregressive training/inference distribution mismatch | If reference-prefix recovery plateaus, compare matched-budget student-prefix KD instead of merely repeating the same reference examples |
| [Official Qwen2.5 results](https://qwenlm.github.io/blog/qwen2.5-llm/) and [technical report](https://arxiv.org/abs/2412.15115) | Publish task-specific 3B-Instruct scores, but not a complete reproducible instruct evaluation recipe | Use a pinned standard local protocol for teacher and student; show published scores as references, not identical-protocol reproduction claims |

Calibration tokens, total processed tokens and loss-bearing response tokens are different denominators. Do not present 13M versus 105k as a clean compute or data-efficiency ratio. None of these papers establishes the superiority of our J-space/Gröbner basis; that needs matched controls.

**Process correction:** our adopted [distillation standard](../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md) already proposed 100k → 1M → 10M supervised-token rungs. We implemented the small rung without carrying the deeper progressive models through a demonstrated adequate recovery budget. The missing step was execution and evidence, not another claim that a standard existed.

## 5. Ordered experiment plan

### Phase A — Freeze artifacts and establish trustworthy baselines

Use the original teacher, the preserved eight-block update46 checkpoint, and a separately frozen three-block recovery checkpoint. For the first broad audit, the already-discussed three-block update134 is a prospective fixed choice; do not switch it after seeing new benchmark scores. Later parent selections remain distinct artifacts.

Before new training:

- Authenticate raw and packed checkpoints, tokenizer/chat template, native scale, bases, head projections, polynomial relations, residual routes, bridges, affine corrections and final norm.
- Check full-vocabulary execution/reload parity on existing DEV prompts and generated prefixes; answer-letter agreement alone is insufficient.
- Count total parameters, executable bytes and each retained component. Distinguish eight compressed blocks from eight removed blocks and from a whole-model compression percentage.
- Freeze a benchmark protocol and a separate recovery DEV set. No candidate selection on the new confirmation benchmark.
- Establish local native performance under the same task formatting, lengths, decoding, numerical precision and scoring used for the compact comparison. If native performance is unexpectedly low, diagnose formatting, extraction, truncation and runtime before attributing the gap to compression.

Broader evaluation has two clearly labelled tracks:

1. **Current-range retention:** paired teacher/student at identical supported lengths, with all out-of-range cases explicitly counted. MMLU-Redux v1 is a useful 3,000-case, 30-subject first test. Report original versus corrected-answer policies, ambiguity exclusions, per-subject results and overlap strata. Choice-logit scoring is not instruction-compliance scoring.
2. **Standard/general capability:** pinned task implementations for MMLU-Redux or MMLU-Pro, GSM8K, HumanEval/MBPP, IFEval, and a separate summarization/application suite. Use task-appropriate generation allowances, not a universal 128-token cap. Compact results requiring unsupported context remain **unsupported**, not silently truncated standard scores. Complete this track after context qualification.

Official 3B-Instruct reference scores are MMLU-Redux 64.4, MMLU-Pro 43.7, GSM8K 86.7, HumanEval 74.4, MBPP 72.7 and IFEval strict-prompt 58.2. Published base-model MMLU 65.6 is not the right comparator. Pin dataset revisions, task versions, shots, prompts, extraction and decoding; unspecified author details remain unspecified.

### Phase B — Test recovery-data sufficiency at fixed eight-block geometry

This is the highest-priority new recovery experiment. Do not add another compressed block, refit its basis, change attention, or alter precision during this test.

| Arm | Initialization | Context and objective | Planned additional supervision |
| --- | --- | --- | --- |
| B0 | Preserved eight-block update46 | No new updates | 0; immutable control |
| B1 | Same B0 checkpoint, fresh documented optimizer | Current 416-token admission policy; exact forward KL, T=1, CE=0; train eight compact blocks and final norm; other state fixed | One continuation trajectory, checkpoints near 100k, 300k and 1M supervised tokens |
| B2 | Continue B1 only under the predeclared DEV/learning-curve rule | Same frozen architecture, objective and corpus policy | Extend cumulative additional supervision to 3M, then at most 10M—not automatic extra epochs |

Start with the currently measured LR `2e-5` as a baseline, not an established optimum. Freeze the optimizer, clipping, effective supervised-token batch size, sample ordering and schedule before running. Checkpoint whole optimizer windows; report exact budgets when a complete example crosses a milestone. Save an epoch-zero candidate and prioritize the earliest checkpoint statistically indistinguishable on DEV.

**Data requirements:** inventory and admit enough new, source-group-deduplicated TRAIN material before reserving GPUs. Extend the existing eight-family mixture in frozen proportions for the budget comparison; do not confound more data with a changed task mixture. Count unique examples/groups, input tokens, supervised tokens, repeats and lengths. Preserve complete responses and real EOS; never turn a capped partial response into a completed target. Report examples excluded by the short-context admission policy.

A separate broadened-corpus version may then add substantial ordinary language, instruction following, code, mathematics, structured outputs and numerical/entity-sensitive summaries. Declare whether each record supervises all language tokens or only assistant responses. A 109GB storage inventory is not a count of usable supervised pairs.

Existing model-only checkpoints do not contain resumable Adam state. Starting B1 means a **new recovery optimizer**, identically initialized for matched arms, not exact continuation of the historical optimizer. From B1 onward preserve optimizer, RNG, sampler and token-budget state for real resume.

**Budget decision rule:** use a fixed DEV panel, learning curves and actual per-family outputs. Go from 1M toward 3M/10M if useful generation or teacher divergence is still improving without a material held-out regression. If two successive planned checkpoints provide no useful-generation improvement, inspect errors and run the next targeted ablation; do not declare the compression method impossible or silently repeat the corpus. A failed task screen remains evidence, not an automatic reason to terminate the existing depth-exploration campaign.

### Phase C — Separate length limitations from data quantity

Short contexts may limit recovery; short output caps may also hide recovered capability. These are different hypotheses.

1. **Evaluation-cap diagnostic:** hold weights and prompts fixed; compare 128-token output limits with the largest supported allowance. Grade completion, cap termination and correctness separately. Never exceed the compact model's total positional support. This measures the evaluation restriction, not training improvement.
2. **Context extension:** retain J-space bases, ranks, weights and original positions where possible; extend affine anchor/shift tables using TRAIN-only calibration at longer positions. Validate denominator/bridge/head behavior and short-range replay before recovery. If the basis itself must change, record a separate geometry intervention rather than call it only a length increase.
3. **First longer-context target:** 2,048 total tokens, followed by 4,096 if memory and quality qualify. The application’s 2,048-token input fragments require extra room for template and output; a 2,048-total-token model is not full coverage. A 6,300-token batch prompt plus up to 768 output tokens requires a later 8,192-total-token qualification.
4. Compare the extended model trained only on matched short examples against the same extension trained on long-context material at matched supervised-token budgets. Evaluate common short cases and independent long-case strata separately.

This is not a clean one-factor comparison if anchors, examples and sequence lengths all change together. Preserve those controls rather than claim a pure “longer tokens caused recovery” result. Never clamp positions, silently truncate prompts, or remove hard examples and present the remaining subset as a standard benchmark.

### Phase D — Address student-history mismatch only after the baseline is credible

Use the selected fixed-depth B/C initializer for two matched arms:

- Reference-prefix forward-KL continuation.
- Clean replay plus teacher feedback on student-generated TRAIN histories; start with a declared 75% clean / 25% rollout supervised-token mixture, then change only in a new arm.

Teacher targets must condition on the exact student token prefix. Recompute the native-prefix boundary for that history; do not attach old reference-prefix targets. No hard-label CE on unverified student mistakes. Refresh rollouts at declared intervals and record staleness, generating checkpoint, actual EOS, cap terminations and teacher identity. Report both supervision budget and wall-clock/GPU cost because on-policy collection is more expensive.

Start the comparison at a meaningful 1M additional supervised-token budget after the disposable systems pilot; the old 24k-token rollout proposal is a shakedown, not the main recovery claim. Use deterministic evaluation and a separately declared rollout sampling policy. Run a second and third matched training seed for the winning treatment before claiming reliable recovery; checkpoint counts are not replicate counts.

### Phase E — Apply the successful recovery recipe across depth

Only after fixed-depth learning curves identify an effective procedure, apply it to further compression:

- Single-new-block fitting sees the **actual compressed prefix**, with projected teacher feature targets and output-aware look-ahead where affordable.
- Follow with joint recovery of the compressed stack against the original frozen teacher. Do not replace the teacher with the previous compressed model without a separate experiment.
- At major depth milestones (8, 16, 24 and 36), perform a substantial global recovery and broader DEV audit. A token budget sufficient for three blocks is not assumed sufficient for 36.
- Preserve all candidate artifacts and distinguish the exploratory depth chain from the best recovered deployable checkpoint. Poor quality can motivate recovery or a parallel diagnostic; it must not silently cancel the user's full-depth exploration goal.
- Local layer work can run in parallel only when its declared inputs are independent. Global sequential recovery has real dependencies; independently good layers are not assumed composable.

### Phase F — Attribute the basis and optimize the recovered execution

Once the recovery pipeline is validated, run matched architecture/budget controls: functional J-space basis versus an activation/PCA basis and a matched random basis. Keep deployed dimensions and recovery budgets equal. This is an attribution experiment, not a decision to replace J-space with average activations.

For the Gröbner component, record the monomial order, fitted relations, normal-form/remainder errors on held-out tasks, cross-task basis redundancy, rank/condition diagnostics, and actual operations eliminated by compilation. Compare relations retained versus removed at a matched parameter budget. Do not count duplicate basis elements as independent compression capacity.

Keep ABLATION32 as a separate basis-selection study with its original random-removal control. Revisit quadratic attention only after a recovered softmax comparator exists; use a one-block matched test before expansion. The stopped retry implementation is reusable preparation, not a completed experiment.

Then optimize packing/fusion, cache support and kernels without changing model quality. A first end-to-end profile should happen earlier to locate costs, but extensive kernel work follows an identified quality-preserving structure. Measure isolated same-hardware teacher/student prefill, cached decode, batch-1 latency and useful-task throughput at matched input/output lengths. Include embedding/readout, bridges, retained residual paths, host transfers, runtime state and KV memory. Distinguish warm inference from compilation, loading and concurrent-training interference.

## 6. Evidence and acceptance rules

Separate three verdicts:

1. **Exploratory progress:** completed run, truthful relative metrics, learning-curve movement and preserved artifacts. A small sample can establish a useful lead without being a deployment certificate.
2. **Confirmed retention:** a frozen candidate evaluated once on independent, adequately sized task/source groups, with paired confidence intervals against the native teacher. Use the user's −5pp margin on prespecified primary task metrics. For noninferiority, require the lower paired confidence bound to exceed −5pp; an interval crossing the margin means inconclusive, not automatically failure. Prespecify family weights, co-primary endpoints and multiple-comparison handling. Estimate sample size from paired discordance and desired power before opening confirmation outcomes; do not declare 16, 256 or 1,000 cases universally sufficient.
3. **Deployment suitability:** confirmed retention plus adequate absolute usefulness for the intended application and measured size/speed benefits. A weak teacher is not evidence of compression failure, but matching an unusable teacher is not application readiness either.

Report per benchmark/task: teacher score, student score, paired change, interval, N, unsupported/capped cases, overlap exclusions, seed variation and metric definition. Do not hide a serious task regression in a large pooled MC total. A six-case canary is diagnostic rather than a hard statistical 5pp gate. Maintain deterministic parser/code/math checks and a separate semantic panel for summaries and open-ended tasks.

Use one pinned judge/version/rubric, blind the model identity and pair ordering, preserve raw outputs, and reuse identical judgments only when the judging context is identical. Rejudge a fixed subset to quantify evaluator noise; manually review disagreements and number/entity errors. Keep `unavailable` separate from `incorrect`. If relative pairwise judging changes context, repeated teacher scores are repeated measurements, not a new teacher baseline.

All new test/benchmark groups remain unavailable to training and basis fitting. Audit exact duplicates and documented near-duplicates across the entire ancestry, including public task TRAIN data and derived summaries. Report residual contamination uncertainty; do not claim the original pretrained teacher never saw public benchmarks. If exposed DEV is used to design new training, retain it as exposed DEV and use genuinely independent confirmation—not relabelled historical tests.

## 7. NAL, resource policy and implementation order

Reuse the [NAL adaptive-recovery plan](nal-adaptive-recovery-plan.md) and existing loss/data/ledger primitives. Do not add another generic scheduler. Minimum work before scaling:

1. One versioned study specification covering model, geometry, corpus, masks, budgets, seeds, endpoints and prospective decisions.
2. A resumable Qwen recovery session preserving CPU Adam masters/moments, model state, RNG, sampler position and committed optimizer-window cursor.
3. Phase separation for data admission, disposable pilot, training, generation, CPU grading and final aggregation. GPU phases run only inside `inference-ctl lease run`; CPU preparation/waiting/judging hold no lease.
4. Stream immutable data/teacher states from disk with bounded prefetch and length bucketing. Verify numerical value/gradient equivalence before block-diagonal sequence packing; packing is not permission for cross-example attention or changed position semantics.
5. Qualify longer contexts and eight-block recovery with real peak memory/update measurements before selecting cards. Keep BF16 execution/FP32 optimizer state for the initial comparison; do not bundle quantization into a recovery ablation.
6. Allocate resources by completed evidence: one main fixed-depth recovery study and, where independently reservable, one evaluation/data job. Existing ABLATION and progressive runs are separately identified, not hidden competing recovery arms.

Measure teacher capture, transfer, optimizer, evaluation and grading costs separately. The observed three-block full pass spent about 47 minutes in training; this is not an ETA for eight blocks, longer contexts, rollout collection or 10M tokens. Derive those estimates from the next real systems pilot and expose the cost before scaling.

## 8. First deliverables after review

1. A frozen baseline report for native / three-block / eight-block models, including range coverage and an honest published-reference comparison.
2. An admitted larger TRAIN corpus and separate DEV/confirmation manifests, with actual unique supervision counts—not an unverified disk-size claim.
3. A real interrupted/resumed training lifecycle and length/memory pilot under reservation.
4. The eight-block budget curve through 1M additional supervised tokens, followed by the declared decision on more data, longer contexts or student-history training.
5. Only then: propagate the demonstrated recovery recipe through depth and publish an end-to-end quality/size/speed frontier.

The principal change is **stop treating a cheap pilot as the ceiling of the technique**. Establish recovery sufficiency, distinguish training limitations from evaluation limitations, and judge compression by paired retained capability before expanding the architecture again.
