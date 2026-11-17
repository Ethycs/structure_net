# Qwen compression recovery: agent execution plan

Date: 2026-11-17. Status: **planned; this document launches no agents or GPU jobs**.

## Objective and decisions

Recover useful generation in the J-space / Gröbner-inspired compact-plus-residual Qwen2.5-3B-Instruct model, then carry the demonstrated recipe through all 36 blocks. Preserve the user's allowance of up to 5 percentage points of accuracy loss. Approximately 38% smaller and 1.4× faster remain whole-model deployment targets, not established results.

This execution breakdown builds on the [recovery reset plan](jspace-qwen-compression-recovery-reset-plan.md) and [NAL integration plan](nal-adaptive-recovery-plan.md). It supersedes their proposed first-study ordering: diagnose the 10→11-block discontinuity; qualify context support; compare substantial reference-history recovery with student-history recovery at fixed eight-block geometry. The older three-block, ~24k-rollout pilot is not the scientific budget for this study. Historical frozen campaigns retain their own contracts.

Evidence motivating the change:

- The observed progressive MC screen stayed at native 195/256 through ten compressed blocks, then fell to 168 at eleven and 122 at sixteen. More depth under the existing small recovery budget is exploration, not proof of an adequate recovery recipe.
- Separate three-block recovery on 105,089 response/EOS tokens improved useful completions from 0 to 4/16 and reliable generations from 4 to 10/16. These are exposed DEV diagnostics, not a broad accuracy certificate.
- Current local fitting uses response-position feature MSE; joint recovery already trains all compact blocks and final norm with output distillation. There is no need to rediscover or reimplement joint training.
- The chain grows backward through a compact suffix. Its native-prefix cache is legitimate on unchanged reference histories. The new block changes the input received by the inherited compact suffix; free generation introduces a separate token-history distribution shift.
- Current compact exports support 416 total positions. Output caps and positional support are different limitations.

## Ownership: four agents plus one integration owner

These are substantial implementation slices, not four independent experiment campaigns. Agents may develop in parallel after the lead freezes interfaces. Only the lead launches scientific GPU work. No new generic scheduler, registry, loss library or optimizer.

### Lead — StudyIntegrator

**Own:** immutable study specification, shared artifact schemas, NAL phase wiring, reservation launch, integration checks, experiment decisions, final report and selective commits.

Reuse `src/neural_architecture_lab/{distillation_recovery,distillation_data,distillation_training}.py`, the existing runner/result ledger, and the qualified Qwen CPU-offloaded optimizer. The current repository has these primitives; the additional campaign/session modules described by the older NAL plan are not assumed implemented merely because they were documented.

Before branching:

1. Freeze teacher revision, tokenizer/template, source snapshot, baseline hashes and the eight-block initializer below. Record inherited training and exposed evaluation groups.
2. Declare shared schemas in one lead-owned module or specification, extending existing types only where needed. No agent independently invents incompatible manifests.
3. Give each agent exclusive paths and its acceptance criteria. Any shared existing-module change belongs to the lead and requires caller/reference review first.
4. Record active producer snapshots. Implement in an isolated, recorded source checkout if shared edits would invalidate active jobs; do not rewrite pinned producers or unrelated user changes.

During execution: integrate bounded slices, run their focused checks once after integration, then run one discarded systems pilot under a lease. Preserve failures rather than silently rerun training. Publish completed checkpoints, next phase, measured resource costs and quality separately.

### Agent 1 — CollapseDiagnostics

**Question:** what causes the first substantial quality loss when the eleventh block is introduced?

**Own:** a thin diagnostic adapter, proposed `experiments/structure_net/jspace_3b_recovery_diagnostics.py`, and its focused tests. Read model/geometry code; do not edit it. Use the preserved ten/eleven-block snapshots rather than an ever-changing latest checkpoint.

Deliver:

- An authenticated checkpoint/state inventory distinguishing local-fit, pre-joint and post-joint states. Missing snapshots are explicitly missing, not reconstructed by assuming current weights equal historical weights.
- A fixed TRAIN/DEV diagnostic panel spanning short QA, math, code, instructions and summaries. Existing exposed DEV is acceptable for diagnosis, never relabelled independent TEST.
- Paired interventions feeding the same frozen downstream suffix either native or compressed boundary activations. Separate insertion effects from subsequent joint changes; identify the exact block and tail checkpoint in every row. With 36 blocks, ten compressed blocks are native indices 26–35; the eleventh adds index 25.
- Boundary reconstruction and next-token teacher divergence, broken down by task and position; a before/after joint update summary. Do not infer a bug from a large parameter update or a causal positional defect from a correlation alone.
- On saved/generated histories: divergence and teacher probability before repetition onset versus inside a loop; natural EOS, clean cap exhaustion and degeneracy separated. Never reuse reference-history teacher targets for changed histories.
- A fixed-weight output-cap diagnostic within supported total positions, with exclusions reported and no position clamping.

**Acceptance:** distinguish an interface/initialization hypothesis from recovery drift and token-history mismatch using controlled contrasts. The report may remain inconclusive. It must not claim all three are proven or label truncated output as incorrect reasoning by definition.

Conditional follow-on owned by this agent: if interface error dominates, implement a matched local-fit arm combining normalized feature reconstruction with teacher-output KL through the frozen inherited compact suffix. Frozen suffix weights must still transmit input gradients. This is a new, separately named intervention; it does not silently enter the main two-arm comparison. No new vocabulary probe is needed if the actual suffix is affordable.

### Agent 2 — CorpusAndEvidence

**Question:** can we train and evaluate against an adequately diverse, uncontaminated, reproducible distribution?

**Own:** corpus admission and evaluation adapters, proposed `jspace_3b_recovery_study_data.py` and `jspace_3b_recovery_study_eval.py` under `experiments/structure_net/`, plus their tests. Does not own model execution, shared NAL primitives or geometry.

Deliver:

- Immutable TRAIN, recovery-DEV and sealed confirmation manifests with source-group/derivative lineage. Admit enough distinct complete supervised responses for the 1M-token control before GPU training. Inventory actual unique input/supervised tokens, length strata and exclusions; disk size is not evidence of usable examples.
- An explicit mixture covering code, mathematics, ordinary instruction/language and numerical/entity-sensitive summaries. Both arms draw from the same admitted prompt population and family policy. Any shift from the historical corpus is documented and shared by both arms.
- Training access restricted to TRAIN. Split checks cover basis/anchor calibration, all recovery ancestry and derived records. Benchmark TEST answers never become recovery targets. Report unresolved near-duplicate/pretraining contamination uncertainty.
- Correct template boundaries and actual EOS handling. Length-limited reference completions are not stored as complete targets. Rollout cap endings are allowed as explicitly partial histories for teacher-feedback KD, without inventing an EOS.
- A broader paired DEV panel: executable code tests in a sandbox, mathematical answer checks, instruction constraints, and a blinded pinned semantic rubric for summaries/open-ended tasks. Report task success and useful completed tasks separately from natural termination and repetition.
- A sealed confirmation protocol with paired intervals and the 5pp noninferiority margin on prespecified task-accuracy endpoints. Determine sample size from paired discordance/power rather than declaring 16 or 256 cases sufficient. Pin benchmark implementations, output budgets, exclusion policy, family weights and judge version before results.

**Acceptance:** every trained/scored row traces to a checkpoint and a declared split; unsupported, unavailable and incorrect remain separate. Identical teacher responses and judging contexts are reused where valid; teacher judgment variability is measured rather than mistaken for model change. Published Qwen scores remain references unless the protocol is actually reproduced.

### Agent 3 — RecoveryRuntime

**Question:** can the same compact model receive fresh teacher feedback on its own histories, with correct resumable optimization?

**Own:** Qwen persistent session/rollout adapter, proposed `experiments/structure_net/jspace_3b_recovery_session.py`, and focused tests. Reuse `ResponseModel`, CPU-gradient Adam, exact hidden-state teacher targets and NAL chunked losses. The lead owns any shared changes to those components.

Deliver:

- One persistent optimizer across clean and rollout rounds; no reset at each refresh. Both arms begin with a fresh, identically configured optimizer because historical selected checkpoints are model-only.
- Seeded categorical TRAIN rollout generation, with exact token IDs, generating checkpoint/update, stop reason, template, sampling policy and RNG state. Greedy deployment evaluation remains separate.
- Original-teacher scoring on the exact student history, including the native-prefix boundary needed by the compact suffix. Capture hidden states and use the original readout; do not create full-vocabulary disk banks or substitute a stale reference boundary.
- Full-vocabulary forward KL, temperature 1, no hard-label CE on student mistakes. NAL already supplies token-chunked backward; its current JSD support is not a reason to change the main experiment's objective.
- Whole-window atomic checkpoints containing model, named CPU optimizer state, scheduler, sampler, mixture/rollout state, global token/update counts, source hashes and all RNGs. Preserve the prior committed checkpoint until the new one authenticates.
- Resume parity and stale-rollout enforcement. Initial correctness pilot refreshes before each optimizer window; any more efficient bounded-staleness policy is declared and measured before the scientific run, not silently introduced midway.
- Bounded host memory, shard handles and prefetch; teacher/student residency can alternate. Record capture, rollout, transfer, update and checkpoint time separately.

**Acceptance:** interrupted/resumed execution consumes the same next examples and preserves the declared numerical trajectory within frozen tolerances; correct teacher/history binding; complete gradients and actual learned-weight changes; unchanged frozen teacher/geometry; real outputs rather than only a not-throw test.

Do not add PPO/GRPO, a reward model, repetition penalties, synthetic positive labels, or top-k target approximations to this first comparison.

### Agent 4 — ContextGeometry

**Question:** can longer histories execute correctly without silently changing the compact model's existing function?

**Own:** explicit context-extension/export qualification adapter, proposed `experiments/structure_net/jspace_3b_context_qualification.py`, plus tests. No changes to active affine-block/stack producers. Coordinate any unavoidable shared edits through the lead.

Deliver:

- Inventory position-dependent anchors, shifts, normalizers and bridge tables; identify every length-dependent component, not only the model configuration field.
- TRAIN-only calibration extension to **2,048 total positions**, preserving the original rows and bases where mathematically valid. No clamping, fabricated anchor rows or silently refitted bases.
- A new immutable geometry artifact with short-range replay parity and finite, correctly masked long-range execution. If extending support requires changing the basis, return a separately identified geometry intervention; never call it a pure context extension.
- Peak-memory and timing profile for prefill, rollout, teacher scoring and updates, including host offload. These measurements run only through the lead's reserved pilot.
- The largest supported per-prompt output allowance. A 2,048-total-token qualification does not cover a 2,048-token input plus its template/output, or the application's 6,300-token prompts. Those require subsequent 4,096/8,192 qualification.

**Acceptance:** supported and unsupported lengths are explicit; both scientific arms load identical extended geometry; short-range output/gradient checks pass at declared tolerances; no quality or speed claim follows merely from longer tensors fitting memory.

## Shared contracts frozen before branching

Reuse existing data types where compatible. The lead defines only the missing fields, and each consumer rejects a mismatched fingerprint.

| Artifact | Required identity / semantics | Producer → consumers |
| --- | --- | --- |
| Study specification | teacher/tokenizer/template, initial model and geometry, source snapshot, arms, seeds, objective, budgets, prospective selection rules | Lead → all |
| Corpus manifest | split, source group/derivative ancestry, token IDs, complete/partial policy, lengths, family, immutable shard hashes | CorpusAndEvidence → diagnostics/runtime/evaluation |
| Geometry artifact | basis/anchor/bridge hashes, original and extended support, calibration TRAIN lineage, parity evidence | ContextGeometry → diagnostics/runtime |
| Rollout record | exact history, generating state/update, sampling/RNG, EOS or cap, round/presentation ID | RecoveryRuntime → teacher scorer/evaluator |
| Teacher target | exact history/predicting-position hashes, native boundary and normalized teacher hidden, teacher/readout identity | RecoveryRuntime → loss provider |
| Checkpoint/phase receipt | parent identities, actual committed budgets, optimizer/RNG/cursor, lease, output hashes and execution status | Runtime/Lead → NAL ledger/evaluator |
| Evaluation row | checkpoint, example/group, raw response, decoding, grader version, completion and correctness, missing-evidence status | CorpusAndEvidence → Lead |

## Execution waves and gates

### Wave 0 — Lead preparation

Freeze shared contracts and ownership. Record immutable source dependencies and artifact paths. Reconcile the already-written NAL primitives with missing execution features; do not implement the old plan wholesale.

### Wave 1 — Parallel implementation, CPU first

Launch the four agents in one batch. Diagnostics and corpus work can inspect existing artifacts immediately. Runtime and context work consume the frozen interfaces, not changing ad hoc manifests. No agent waits idly for a GPU or independently acquires one.

Each agent skips mid-flight builds/tests/formatters and hands off its implementation plus relevant narrow checks. The lead integrates once, runs the focused checks, and addresses reported failures without rerunning broad suites. Tests cover consumer-visible masking, state/history identity, resume, positional semantics and grading behavior—not wiring/source text.

### Wave 2 — Reserved systems and diagnostic qualification

The lead schedules available cards non-preemptively through `inference-ctl lease run`; reservation readiness, not observed free VRAM, authorizes execution. Prefer the 3060 for the main BF16 recovery profile, but do not assume a second card fits until profiled. CPU preparation, grading and waiting have CUDA hidden and hold no GPU lease.

Run a disposable real pilot exercising generation, exact teacher scoring, two optimizer windows, interrupted/resumed state and fresh export/reload. Validate both record types and short/extended lengths. Its examples are TRAIN and its updates never become the scientific initializer. Compare the resumed path with an uninterrupted path; measure memory and phase costs. Then run the frozen 10→11 diagnostic. Reuse compatible captures rather than repeat teacher passes.

If a correctness defect is found, fix and qualify it before scientific training. If length extension is not qualified, report the reason and do not spend the main budget while silently claiming long-context recovery. A separately labelled current-range diagnostic remains permissible.

Existing progressive and ABLATION32 campaigns remain distinct and preserved. This plan does not authorize killing them, changing their producers, or turning a failed quality screen into an implicit cancellation of the all36 goal.

### Wave 3 — Main fixed-depth comparison

Initializer: preserved eight-block update46:
`/data/structure_net_experiments/jspace_3b_full_progressive/20261116_v1/blocks_08/recovery/run/checkpoints/update_0046.pt`

SHA256: `9d2e43b25aa07bc34f549cc91b6e95258bf08721733990ff10c8c51704de6958`.

Both arms use the same qualified context-extension artifact, learned-weight initializer, native teacher, trainable set, precision, optimizer recipe and corpus population. Preserve the unchanged original checkpoint as the zero-update control.

| Arm | Supervision | Additional supervised-token targets |
| --- | --- | --- |
| A: ReferenceRecovery | 100% complete clean/reference-history teacher KD | 100k, 300k, 1M |
| B: StudentHistoryRecovery | Target 75% clean / 25% fresh student-history teacher KD by loss-bearing tokens | 100k, 300k, 1M |

The 25% mixture is a declared initial design, not a proven optimum. Hold forward KL/T=1/CE=0 fixed to isolate history distribution. Start from the previously used 2e-5 LR unless the discarded numerical pilot exposes instability; any change is frozen and applied to both arms before the study. Sampling temperature and filtering are likewise frozen before collection, not optimized on confirmation results.

Use one initial matched seed pair. Match target token budget, effective supervised-token batch size and prompt-family policy. Whole examples/windows may cross a target: freeze a maximum one-window overshoot, report actual budgets and realized mixture, and never call unequal exposure exactly matched. Count repeated clean examples, rollout refreshes, unique prompts, generated tokens and physical GPU time separately. No extra epochs or 3M/10M extension occurs without the predeclared DEV learning-curve decision.

Generate/grade candidates at zero, 100k, 300k and 1M. Keep both arms on one hardware/precision profile; CPU agent parallelism does not imply simultaneous multi-card training. Alternate arms at committed checkpoints if resource scheduling requires it, without resetting optimizer state. Report quality versus supervision and quality versus wall-clock.

### Wave 4 — Conditional mechanisms, not an unconditional grid

- **Interface damage dominates:** CollapseDiagnostics tests output-aware local fitting at the 10→11 transition against the existing local-MSE baseline, with equal data and subsequent recovery budgets. If this changes the main initializer, start a new named study rather than splice it into A/B.
- **Student-history arm helps, but loop tails consume the budget:** compare a fixed rollout horizon against a ShortOPD-inspired repetition-gated horizon, using the same objective and maximum context. Do not censor evaluation outputs, mask away bad training tokens by default, or treat every repeated code pattern as degeneration. Use teacher feedback to establish whether loop tails really provide little correction. The paper's 512-token detector window cannot be copied into our historical 128-token cap unchanged.
- **Trajectory mismatch is addressed but loss choice remains suspect:** run a separately named forward-KL versus generalized-JSD comparison. Do not conflate this with the first history comparison.
- **Both arms improve through 1M:** extend the informative trajectory toward 3M, then 10M under a frozen DEV rule; preserve unique-data/replay accounting.
- **Neither improves:** use the diagnostic results to select one next intervention—optimization/target repair or functional sensitivity-based redistribution of retained ranks/residual capacity. Do not declare all J-space compression impossible, and do not launch every option at once.

All these are conditional work packages, not permission for agents to expand scope independently.

### Wave 5 — Confirmation and return to full depth

If a treatment is promising on DEV, run two additional matched seed pairs to test repeatability. Freeze candidate selection before opening the sealed confirmation set. Report paired task accuracy, useful generation, per-family differences and intervals; an interval crossing the −5pp margin is inconclusive, not proof of retention or universal failure. A screen of 16 generations remains diagnostic.

Carry the demonstrated recipe through subsequent compressed depths with substantial joint recovery at declared milestones, then all36. Preserve the exploratory chain as separate evidence; do not conflate it with the recovered candidate. Finish with isolated, same-hardware whole-model bytes, prefill, cached decode where supported, and latency/throughput at matched lengths. Packing/kernel work must preserve numerical behavior and counts every retained component.

## Deliverables and decision ownership

The lead reports, in order:

1. Frozen study/ownership/source contracts and admitted data inventory.
2. 10→11 diagnosis, context qualification and measured systems-pilot cost.
3. A/B learning curves and representative failures, with actual budgets and per-phase runtime.
4. One justified next decision: continue, apply a targeted treatment, redistribute capacity, or confirm the winning recipe.
5. Full-depth quality/size/speed frontier only after that recipe has actually been applied.

Neither a subagent completion message nor a saved checkpoint is a quality claim. Agent work is verified at integration; scientific claims require actual authenticated evaluation.

## Research basis

- [ShortOPD, v2](https://arxiv.org/pdf/2607.13124v2): Qwen3 depth/width/hybrid generation recovery; student histories and repetition-aware rollout budgets. The width result is materially better than fixed-history KD but still below the teacher. Not a reproduction of our J-space architecture or a promise of the 5pp target.
- [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649): teacher feedback on student histories, with trajectory choice separate from divergence choice.
- [Per-Matrix Optimality Is Not Enough](https://aclanthology.org/2026.findings-emnlp.251.pdf): local, downstream-sensitive and global optimization address different error scales; perplexity gains do not establish retained generation accuracy.

The execution change is deliberately narrow: **diagnose the first failure, qualify usable lengths, compare two substantial recovery loops, then propagate evidence—not multiply independent small experiments.**

## Execution log — 2026-11-17

The four implementation slices were launched together with exclusive ownership. The shared study is `/data/structure_net_experiments/jspace_3b_recovery_reset/20261117_v1/spec.json`; it starts from the selected eight-block `update_0046.pt`, not a newly recombined layer set. Main integrates the adapters through `jspace_3b_recovery_workflow.py`: the existing NAL runner dispatches CPU control jobs, and GPU children acquire their own `inference-ctl lease run` reservations. Execution receipts are not quality scores.

Initial CPU findings: selected block9→block10 preserves all 259 inherited tensors and the norm; selected block10→block11 prejoint preserves all 288 inherited tensors and the norm. This rules out a dropped selected-parent checkpoint in that handoff. Saved 22-generation panels contain 18 capped/4 EOS outputs at block10 and 16 capped/6 EOS at block11. Cap counts alone neither establish correctness nor identify the cause of regression.

Context qualification extends TRAIN-observed affine position rows to 2048 with the original fitted bases and learned weights held fixed. Existing rows must remain exact; the real pilot checks short-range output/input-gradient/parameter-gradient parity and full-range causal masking/finiteness. Block checkpointing bounds pilot activation memory. Qualified support is not evidence of long-task quality.

The runtime freezes a common supervised-token window target derived from 16 times the admitted mean complete TRAIN response length. Sixteen is a nominal clean-equivalent batch size, not a claim that both arms use sixteen histories per update. Complete-response overshoot is recorded, gradients are normalized by actual supervised tokens, and optimizer/sampler/RNG state survives cumulative milestones. Native DEV generations may be reused only under identical authenticated model, prompts, template and decoding settings; teacher training targets must follow the actual student history.

The 3060 remains assigned to the existing progressive campaign (block18 pilot observed during integration). The new qualification profile uses the free RTX2060 SUPER under a 7680MiB reservation, with lower worker limits. An unrelated ABLATION32 controller ended with `execution_failed` before `learned_initial_evaluation_prepare` started its producer; its receipt and partial scientific artifacts remain preserved, not silently restarted.

Integration verification: 72 focused CPU tests passed (32 geometry/diagnostics/orchestration, 17 session/recovery, 23 corpus/evaluation). Use locked Pixi `-e ncm` and `pytest --confcutdir=tests/structure_net`; the unrelated repository-wide conftest requires SciPy absent from this environment. The actual NAL CPU runner authenticated and adopted prepared data, evaluation, and context artifacts. Its generic `Accuracy: 0.00%` console field is not a model measurement; these jobs report phase time and `scientific_claim=false`.

Actual admission: 1,100,165 complete TRAIN supervised tokens, with 275,066 math, 275,076 code, 275,015 instruction and 275,008 summary tokens; 512 independent DEV groups; 3,140 independently grouped sealed confirmation cases; 24 separate TRAIN-only 2,048-token calibration prefixes. Confirmation is byte-authenticated, not decoded by the admission checker. The 3,140-case count is a normal-approximation planning target, not guaranteed 80% power for the more conservative Hoeffding interval; the manifest records this mismatch and conclusions use the actual interval.

Readiness controls passed: the pinned `codex_sdk/gpt-5.6-sol` judge correctly separated a constructed correct/incorrect arithmetic pair and returned the expected model identity. The existing bubblewrap executable grader passed its arithmetic/isolation control with `/data`, rabbit's home, `/opt`, and `/root` hidden. Neither control is model-quality evidence.

Implementation commit: `6082e9740e`; 11 additional native-streaming prerequisite tests passed (83 focused checks total). The existing uncommitted MMLU-Redux runner and its tests were included deliberately because the recovery runtime imports its `host_weights` context manager; unrelated benchmark-coverage work remains uncommitted.

The real reserved context pilot passed on the RTX2060 SUPER. Existing-range output and input/parameter gradients were exact; 2048-position gradients were finite; causal and interior-key masking checks passed. Suffix-only full-range forward/backward took 4.213s, with 4,225,467,904 peak allocated bytes and 6,276,775,936 peak reserved bytes across the pilot. Its internal duration was 54.15s; the whole NAL phase took 216.5s including startup and authentication. These are systems measurements, not recovery accuracy or whole-model decoding speed. The mixed-history runtime/resume pilot is the next independent gate.

The subsequent v1 runtime pilot failed before optimizer construction: the generic compiler constructor recreated eight saved BF16 polynomial-bias parameters as FP32. The original preservation predicate used `torch.equal`, which checks values but permits different dtypes; thus v1's within-loader parity did not establish exact checkpoint precision. No recovery updates occurred. The strict loader now restores saved tensors with `load_state_dict(assign=True)` and preservation checks require identical dtype, shape and values; FP32 geometry is not globally cast down. All 16 focused context tests, including the new dtype regression, passed.

V1 artifacts remain untouched. Corrected execution is isolated under `20261117_v2_precision/spec.json`, retaining the same initializer, seed, data-selection policy and training budgets. Corpus admission and context qualification are rerun under the corrected frozen source rather than relabeling the old evidence as valid.

V2 passed both the corrected context gate and the real streamed-prefix recovery pilot. Two disposable optimizer updates consumed 463 response tokens (306 reference, 157 sampled); actual BF16 weights changed. A fresh process reproduced model/optimizer state, RNG, next examples, masks and teacher-history bindings exactly across 306 compared model/optimizer payloads. This is systems qualification, not new task accuracy.

Measured baseline pilot costs were 254.39s for rollout, including 245.07s of native-prefix execution/streaming; 23.16s for teacher capture; 12.31s for backward; 5.49s for Adam; and 174.49s for the two measured checkpoint commits. The raw linear token extrapolation is not a reliable full-study ETA: the scientific window targets 1,017 supervised tokens, unlike the four-example pilot, so fixed checkpoint costs are amortized differently.

Before committing the long study to that streaming cost, a separate `runtime_resident` profile is testing the already-implemented GPU-resident native prefix under the same SUPER reservation. The NAL controller now pins a runtime root and residency mode, preventing one profile's successful pilot from admitting another. Six focused orchestration tests passed. Data/geometry and objective remain unchanged; the original streamed pilot and its artifacts remain intact.

### Full-context residency qualification

The fully resident native prefix passed the short rollout/resume pilot but failed
on a real 2,047-token TRAIN history: the native attention softmax could not allocate
another 256 MiB. It is not admitted for recovery on the full corpus.

The replacement profile is `runtime_resident24`: 24 of the 28 native prefix blocks
remain on the SUPER during rollout; the remaining four stream from their original
CPU storage. Both paths now use the same `ExactHistoryTeacher.forward` arithmetic.
The runtime plan records `native_prefix=resident` and
`resident_prefix_layers=24`; its distinct root and plan fingerprint isolate it
from the earlier full-resident pilot. Residency is released before teacher target
capture and backward, including when reservation admission raises.

The disposable pilot now requires bit-exact short and full-context next-token
logits against streamed execution before saving checkpoints or taking an optimizer
step. This closes the short-rollout qualification gap rather than shortening the
context or relaxing the memory guard. Runtime/workflow regression checks passed:
27 tests. The new reserved hardware pilot is in progress; no quality improvement
or full-corpus speedup is claimed from these systems checks.

The 24-block residency hardware qualification also failed at full-context native
attention softmax, before any optimizer update. Its failed artifacts are retained.
`runtime_resident16` / `workflow_resident16` now test 16 resident and 12 streamed
native-prefix blocks under the same reservation, precision, context, and objective.
The smaller resident bank is a workspace-headroom decision, not a model change.
