# Native-query language distillation followed by vetted-summary training

Date: 2026-10-23. EXPLORATORY `ncm-native-query-neighborhood-training-v1`.
Known evidence: the actual full-native NCM passed its two-update lifecycle with
8,637,696new parameters, frozen originals/books and exact free-output reload.
Its two-update text is unusable; the pass is not task competence. The new main
teacher drafts scored111/128and their single revisions are still running. The
old five-block0/64transfer failure and all teacher failures remain unchanged.

This study uses INDIVIDUALLY accepted training labels, not an assertion that the
teacher population meets95%. Admit all111new first-draft positives from the
qualified128-token Terra critique plus all32previously independently accepted
Sonnet training labels. Verify every first judgment, literal evidence replay,
source identity, tokenizer round trip and both128token limits. No semantic
regeneration or choosing the better draft/revision. All143labels are fixed
before this student's full training. Record each label's original judge/profile;
the older32use the stricter approximately32-word salience budget. Original full-
sample teacher-pipeline gates remain separate and their failures stay failures.
This explicitly separate accepted-label study does not rescue them or admit
scaled generation under a failed parent protocol. Student evaluation stays on
ALL64original source-disjoint development cases, never the selected labels.

Language data: deterministic hash selections from the8192approved public training
sources. Use1024training and64diagnostic groups with at least255native tokens.
Context is first128native tokens (including native BOS); target is the following
127tokens plus an explicit EOS window boundary. Keep full-source IDs/hashes and
token-window metadata. This is bounded language modeling, not a claim that the
window is a valid summary. Context excludes the target token window. Exclude all138new
teacher-pipeline groups and32older label groups from language selections; also
exclude all576concept-bank preparation groups from language diagnostic data.
Use independent hash prefixes for language train/diagnostic; all groups and
source hashes disjoint across those splits and summary labels. Verify every
source against approved public data and original model-development split.
Keep final256text sealed. Raw language context frame128, summary prompt frame768,
word frame128; this prototype does not establish the full6300-token application.

Use the same intact28-block3B native-query NCM with fixed five256-entry native
residual books,4/8/16/32/64codes then128parallel words, rank8query attention LoRA,
rank16zero-output current-position hint and native word head. Fresh init seed42
from original pretrained weights; randomize only new heads/LoRAA, zero LoRAB/hintB/
level biases. Never initialize from an old failed readout or its optimizer.
All original parameters and codebooks stay frozen. No parent AR/verifier at
inference; every evaluation uses actual five-level generated history.

Precompute source-only frozen KV and sparse native teacher targets on GPU0,
BF16native/FP32new, SDPA, batch2. Teacher sees true previous native words with
query adapters disabled; keep top32probabilities renormalized within that support.
Teacher values are offline constants, not an exact full-vocabulary KL. Store
per-case CPU caches with hash manifest and verify them before training. Cache
placement changes training work, not application prefix-lookup reuse. A batch
contains one task/frame type. Keep actual GPU source KV immutable across levels.

Train1024language updates then1024summary-phase updates, with25%language replay
in phase2. Batch2, accumulation4, AdamW lr1e-4,weight_decay0.01,clip1. First128
updates sample only coarse levels0–4uniformly; thereafter word-level probability
0.5, other levels uniform. Mix source-only actual predicted histories and
conditional-expert earlier histories50/50. Apply target-independent corruption
rates[0.5,0.05,1/30,1/60,0] in both arms; recompute current expert against the
actual preceding noisy history. Experts never enter source cache. Word labels
are native gold tokens, with loss0.5hard/0.5sparse-teacher for language and
0.9hard/0.1teacher for summary. Disable teacher loss on injected raw-window EOS.
Use one weighted gathered-log-prob objective to avoid duplicate dense losses.
Seed streams: init42,sampler2042,history3042,level4042,task5042,corruption6042.

Before full training: CPU objective/gradient/cache/data split contracts; real
GPU cache+teacher preparation; fresh four-update pilot exercising raw/summary
frames, coarse/word losses, immutable/frozen checks, exact checkpoint/generator
reload and full128word activation ceiling. Require7.5GBfree on GPU0; record
expandable-segment allocator. Stop on OOM, nonfinite values, data/hash/cache/
gradient/reload failure. Never silently reduce source/target dimensions.

Raw diagnostic token recovery and training losses are mechanism measurements
only; final fixed checkpoint is the sole model-development endpoint. After all
2048updates, generate and freeze all64complete summaries, count all over-cap/
empty/unreadable outputs as failures. Qualified128-profile Terra, first judgments
preserved; require64valid and>=61whole passes before independent Sonnet. Same
gate on Sonnet. No output repair or semantic retries, no best-checkpoint picking.
If either fails, report actual numerators and continue with a new declared study.
Passing this short-source development gate still requires a separately specified
representative source-disjoint final evaluation before the95%goal is complete.
No speed claim before task quality. Final source-disjoint/long-range work remains.

Roots `20261023_native_query_neighborhood_{data,cache,pilot,train,outputs,grade}_v1`.
Preserve sources, seeds, full configs, cache manifest, raw outputs, checkpoints,
first requests/responses, failure states and conservative aggregate/readback.
