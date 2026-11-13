# Locate effective dense donor interventions

Date: 2026-11-13. ID: `JSPACE-DONOR-LOCALIZATION-01`.
Prospective systems/mechanism pilot, one frozen Qwen2.5-3B-Instruct checkpoint.
The preceding causal-component study stopped because dense final-token patching
at block 24 redirected only 1/88 eligible pairs. Its source/donor first tokens
were already complete colour words, ruling out a leading-whitespace timing
explanation for that result. Prior outcomes remain immutable.

Question: does the failed single-position control reflect patch implementation,
layer depth, or the distribution of task information across question positions?
Prediction: at block 24 replacing question states other than the final token
will redirect more answers than replacing the final token alone. This does not
predict causal concept vectors, J-space sufficiency, or compression.

## Data and masks

Prepare 64 fresh source/donor pairs using the existing closed-world JSON task,
with four examples in each country × original-colour cell. Generate a fixed
pool of 512 evaluation candidates with seed 173; use 128 shakedown candidates
with seed 1000176. Exclude every named source world from the relation and causal
component studies. Select candidates deterministically by case ID, consulting
only tokenization and balance, never model outputs. Donors retain identical
facts and ask about the person assigned to the target country. Select four
token-aligned shakedown pairs separately; they are not quality evidence.

Require source and donor to have equal full prompt-token lengths, equal
question-start token indices, and identical token IDs before the question.
Freeze these IDs/boundaries in the plan. If the fixed candidate pool cannot
fill all 16 cells, stop preparation rather than changing the seed or count.
This is a deliberately length-matched population, not a general summarization
or natural-language benchmark. Anonymous mapping shapes can recur.

At zero-based blocks `[5,11,18,24,29,34,35]`, compare prefill-only replacements:

- `final`: donor's final prompt-position state only;
- `query_suffix`: all donor states from the question start through the
  assistant generation prefix;
- `query_except_final`: the same question suffix, keeping the final source
  position unchanged.

Replace actual BF16 states, preserving the untouched positions exactly. Cached
continuation tokens are not patched. Store complete first-word generated
answers (existing eight-token cap) and first-step logits as separate measures.
No J projection, CAV learning, intervention scaling, or model training occurs.

## Validity and outcome order

1. CPU tests must show terminal-token donor logit equality in a tiny real Qwen,
   full-suffix donor equality at earlier layers with a common prefix, mask
   boundaries, hook cleanup, data balance/selection, and score arithmetic.
2. Under a live GPU lease, use four shakedown pairs for finite computations,
   identity copying, terminal copying, capture roundtrip and memory admission.
3. Capture all 64 clean source/donor pairs and full states at the seven layers.
   Their hidden prefixes must be bitwise identical before the aligned question
   boundary. This audits causality/alignment before full-suffix equivalence is
   assumed. Save suffix states and source/donor first-step logits.
4. For every pair, check a final-block self-copy and final-block donor-copy.
   Self-copy logits must exactly match source logits; donor-copy logits must
   exactly match donor logits. Exact equality is the validity gate (not merely
   equality of argmax). Record generated-answer equality separately.
5. Source and donor complete-answer accuracies must each reach 90%, with at
   least 80% of paired units correct on both. Eligible pairs are fixed by these
   clean outputs. Terminal donor patching must redirect at least 80% of them.
6. Run the frozen seven-layer/three-mask grid on all 64 pairs. Reuse the already
   recorded terminal/final cell exactly once. Full-suffix replacement must
   reproduce donor first-step logits at every layer, given the identical prefix.
   Report and stop on any failed equality/copy invariant; no silent tolerance
   widening or precision fallback.

Primary endpoint: the difference in counterfactual-answer success between
`query_except_final` and `final` at block 24, among clean-correct pairs.
Classify `distributed_question_state_supported` only if validity passes,
the difference is at least 50 percentage points, and the paired two-sided
exact McNemar p-value is at most .05. Otherwise classify
`patching_valid_no_large_distributed_advantage`. Keep `confirmed: false` for
this bounded single-checkpoint pilot. A failed validity/competence gate has a
distinct stopped classification and a null primary endpoint.

Report correct/total counterfactual answers, original-answer retention, other
outputs, Wilson intervals, first-token donor agreement, and full-vocabulary
logit discrepancies. Report all pairs and eligible pairs separately. Mark any
mask/layer reaching 80% eligible counterfactual success as operationally usable;
the earliest final-token layer is a descriptive result from the fixed grid,
not a newly tuned CAV test. Terminal copying is an output/plumbing control, not
evidence that an interpretable intermediate concept has been found.

## Execution

Freeze source, protocol, task/token alignment, parent data, model file hashes,
runtime and root Pixi lockfile before new model outputs. New append-only root:
`data/experiments/jspace_donor_localization/20261113_v1/`.
Prepare/analyze on CPU with CUDA hidden. Capture through
`inference-ctl lease run --allocation 3060=10240 --minutes 60`, root locked
Pixi `ncm`, one GPU worker. Check the lease before CUDA access and every
32 recorded calls. Both data and method invariants fail closed. Preserve
partial records on errors and release the lease on child exit. No service
activation and no scientific retry.

No result here rescues or overwrites the preceding stopped study. A successful
localization would support a new component study at effective positions/layers,
possibly decomposing paired donor-minus-source differences before trying shared
concept vectors. Gröbner relation transport remains a separate later question.
