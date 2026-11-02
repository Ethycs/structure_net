# Native-token readout recovery controls

Date: 2026-10-23. EXPLORATORY, not a rescue of any completed NCM experiment.
Known failures: free-text attachment0/64; source-copy query37/64; original
native parent50/64 and stricter native parent49/64. Broader short teacher labels
passed113/128. Frozen-XSum-decoder data sweeps are closed. The final256 task
cases remain sealed. No control below establishes NCM task success or speed.

## A. Qualify labels with the real output allowance

Hypothesis `ncm-native-budget-label-quality-v1`: a fixed source-grounded teacher
instruction with up to128 stored tokens produces at least122/128 independently
accepted summaries. This replaces the old24-word/48-GPT2-token label constraint;
it does not alter the whole-summary rubric. Teacher is the already authorized
Sonnet alias; independent judge is the qualified Terra alias.

Select138 sources prospectively by SHA256("native-label-audit-v1:" + id) from
the existing8,192 fresh public training sources. Exclude every source/session/
group/hash in the previous expanded audit and its four-case pilot. Original
source partition and public-span proof must verify; retain one group/session/
source hash per case. First ten are systems-only; remaining128 are the fixed
quality sample. No development/final-test sources or old summaries go to teacher.

Ask for a coherent40–70-word summary within128tokens, preserving material
conditions, attribution, modality and negation. Ask for claim/evidence quotes
in the same response, instructing the teacher to check them before returning
one final candidate. Quotes must be literal source spans. This is a single
generation with self-check instructions, not judge-guided candidate selection.
Freeze the exact instruction in code before any request. Parse only JSON/ID/
field structure for retry eligibility. An unsupported claim, nonliteral quote,
long summary or other semantic defect is retained; no semantic retry, clipping,
revision, seed search or case exclusion is permitted.

Teacher requests use two passages, four concurrent workers, temperature0,
max2048 output tokens,180-second timeout and at most three attempts for transport
or schema failures. Pilot first submits one two-passage batch serially, then
four independent two-passage batches concurrently, checking complete coverage, raw replay, literal quotes and
128-token fit. A failure stops this configuration. Before main generation,
require the same producer sources and input hashes as that pilot. Independent
grading first uses the first four teacher-pilot cases as a systems pilot (four valid decisions, not four
semantic passes), then128 cases. Each exact source-summary/rubric/role packet's
first judgment is retained, including semantic negatives and exhausted invalids.
Main requires all128 valid and at least122 whole summaries passing. Report
faithfulness/salience/readability failures and Wilson interval. A data-quality
pass only admits a separate prospective scaled-label/head-training protocol;
it does not qualify unseen labels or establish student accuracy automatically.

Roots `20261023_native_labels_{prepare,pilot,generate,judge_pilot,audit}_v1`.
Never overlap these requests with the native-parent grading producer. Preserve
all raw responses, citations, first-valid attempts, oversize texts and invalids.

## B. Establish native-language head capacity without a text codec

Hypothesis `ncm-native-readout-capacity-v1`: a small frozen-language readout with
trainable interfaces can fit32 previously accepted training summaries and use
source information. This is an autoregressive reference for a later genuinely
coarse-to-fine NCM head, not itself NCM. Its data are training-only and no quality
or efficiency conclusion can be based on this capacity control.

Use the first32 accepted labels in the preserved recipe quality-v2 selection;
replay their original positive judgments and verify public sources. No new
quality filtering. Extract per-token original Llama states after layers1,14,28
under the fixed native source prompt, with no target tokens in the body pass.
Keep the original28-layer body and all its FC weights unchanged. The native
tokenizer must round-trip every target exactly; targets over128 native tokens
are failures, never truncated. Parent source prompt limit768, target limit128.

Readout retains native embeddings, tied word head, final norm and copied blocks
0,7,14,21,27; all their base parameters are frozen. Train only a token-wise
three-layer-feature fusion projection and rank16 LoRA corrections to each
copied attention/MLP projection (alpha16, no dropout). Source features occupy
the prefix; target inputs are BOS followed by previous target tokens. Standard
causal attention prevents future-label leakage. Source padding and positions
must be preserved. No verifier or full-parent call occurs during head rollout.

Before training, focused tiny-model tests must establish causal masking, frozen
native weights, gradient routing, tied word identity and full-versus-cached
readout parity. Then a four-case real GPU0/BF16 pilot extracts features, does
two actual updates at batch2, separately exercises a128-token target for peak
memory, saves/reloads only trainable parameters exactly and repeats greedy
generation. Stop on source/weight drift, OOM, nonfinite values, cache mismatch,
missing gradients or serialization mismatch. Require at least7.5GB GPU0 free
before the original body extraction; do not disturb resident services.

Only after that pilot may the fixed32-case capacity run do1,024 updates,
batch2, accumulation4, AdamW LR1e-4, weight decay0.01, gradient clipping1,
seed42 and independent sampler seed2042. Measure only the final checkpoint:
free-running target-token accuracy at least98% and exact texts at least31/32;
source-shuffled token accuracy must drop at least5 percentage points. Earlier
checkpoints/teacher-forced loss cannot rescue a failed gate. EOS termination
and all overlong/empty outputs remain in denominators. This admits no scaled
student run unless qualified new labels and a separate task protocol exist.

Roots `20261023_native_readout_{pilot,capacity}_v1`. Hash the complete producer,
native weights, labels, original positive raw judgments, source extraction
prompt and runtime. Report actual outcomes conservatively and register/read
back completed studies. Source-disjoint final task95% remains the objective.
