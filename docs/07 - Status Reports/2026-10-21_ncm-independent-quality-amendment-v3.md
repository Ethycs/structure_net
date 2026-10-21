# Verifiable evidence citations for independent summary grading

Date: 2026-10-21. Hypothesis: `llama32-3b-ncm-independent-quality-v1`.
This technical recovery follows the original preregistration and v2 amendment.

Both models completed all 240 sources. The v2 free-quotation judge completed
eight pairs: seven valid and one invalid after exhausted retries. Some quotations
joined turns while omitting speaker labels, violating the literal-evidence
contract. The stage was stopped because the complete strict endpoint was already
impossible. Its raw attempts and interrupted coordinator are retained. The first
three model/source pairs were also inspected and showed unrelated NCM project
outputs; no full quality aggregate was computed. This amendment is therefore a
disclosed technical recovery after limited outcome exposure, not a fresh blind
confirmatory experiment.

Keep the exact frozen sources, 240 outputs per model, anonymous candidate IDs,
orders, primary/secondary judges, fixed audit subset, calibration cases, 0–4
scales, strict pass rule, 95% target, and uncertainty calculations. No training,
model-output regeneration, case selection or quality-threshold changes.

Represent the complete source as its original ordered newline-delimited turns,
including their original speaker labels. Assign `T0`, `T1`, ... IDs. Concatenating
the turn text with newline must recover the source byte-for-byte. Judges cite
these IDs for each material claim. Code validates every ID and saves the exact
original turn text and spanning literal source quotation; it never accepts a
model's reconstructed quotation. Supported claims require at least one cited
turn. Unsupported claims may cite no turn. This validates evidence identity;
entailment remains the independent judge's decision.

Require compact JSON and concise atomic claim text/rationales to avoid spending
generation on duplicated source quotations and formatting. Score every material
claim; brevity must not remove claims. Repeat the same 48-summary calibration
gate with the new evidence schema before its primary judgments. Calibration
thresholds and factual-negative gate remain unchanged.

Use four primary network workers, one secondary worker, never both jobs at once.
Try at most three identical requests for infrastructure/schema failures; never
retry valid quality failures. Stop secondary calibration after an exhausted
invalid pair, recording agreement as unavailable if it cannot calibrate.

Fresh roots: `data/experiments/ncm_quality/20261021_judge_v3/`.
Reuse completed v2 model outputs only after fingerprint, checkpoint, source,
native token decoding and source/hash checks. Recalibrated v3 labels are the
reported descriptive grade; the original v2 complete endpoint is invalid.
Report the recovery, unchanged quality rules, raw result counts and old failures.
No speed claim, no measured production accuracy, and `confirmed:false`.
