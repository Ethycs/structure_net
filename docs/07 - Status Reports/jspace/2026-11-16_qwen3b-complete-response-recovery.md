# Qwen3B complete-response recovery on progressive and all36 students

Study: `JSPACE-QWEN3B-RESPONSE-RECOVERY-01`. Exploratory; follows
[NAL-STD-DISTILLATION v1](../../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md).
The user explicitly instructed execution of the standard on both models.
This is a new recovery study, not an amendment to the old sparse-MC objectives.

## Transition and fixed question

Does complete-response distillation from the original frozen Qwen2.5-3B-Instruct
restore useful generation in the existing compressed students? Preserve the
progressive 16-block checkpoint selected at update144 and all all36 retained
checkpoints through its next update144 boundary. They are different compression depths
and histories, so their comparison is descriptive, not a matched depth ablation.
The progressive old scheduler is stopped before further expansion; its already
leased block17 boundary phase may finish. The old all36 worker is interrupted
only after its update144 checkpoint and DEV record are durable. Retain all old
artifacts. Phi remains paused.

The original model revision is `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
The progressive source is `jspace_3b_progressive36/20261115_v1/stages/blocks_16`,
cut20, selected checkpoint SHA256
`e4bd8882d68eed06242977460a9e86ff48251a940553105a520575873c9316a2`.
All36 source is `jspace_3b_broad_recovery/20261115_v2/all36_joint`.
Initializer decision before any new response-recovery outcome: the newly saved
all36 update144 scored100/256 old MC DEV, versus109/256 for update72. Following
the user's preference for early high-accuracy checkpoints, initialize the new
all36 recovery from **update72**, while preserving update144 as the latest old
training state. This is explicitly old-DEV selection, not a fresh quality claim.
Machine plans authenticate each actual consumed initializer and source plan.
Keep bases, relations, anchors, bridges, native prefix and tied readout frozen;
train the compiled compact blocks and final norm only.

## Corpus and objective

Use a broad task mixture: instruction following and summaries from pinned Dolly
and XSum, factual QA from ARC and BoolQ, commonsense continuation/coreference
from HellaSwag and WinoGrande, arithmetic reasoning from GSM8K and code from
MBPP training data. Use complete answer text, continuation, reasoning or code
as appropriate; do not relabel four sparse answer positions as full responses.
Corpus manifests preserve reference provenance,
tokenizer/template digests, source groups, duplicate exclusions, length-rejection
counts, actual admitted per-family availability and benchmark exclusions.
Short-answer families contribute their available unique examples without
artificial duplication to fill a token quota. XSum references are human summaries, not an
assertion of factual perfection. A normalized exact/shared-text grouping policy
keeps source derivatives in one split. Sealed final benchmark contents must not
enter targets, training, selection or this initial exploratory rung.

Access incident during the superseded two-family v1 preparation: an exclusion
loader decoded `jspace_accuracy_recovery_data/20261114_v1/tasks.json`, a reserved
final512 task file, to collect question text. It did not evaluate model outputs,
but the file contents were accessed, violating the identity-only preparation
boundary. Preserve v1 and its separate correction receipt; do not claim that
final512 remains untouched. The multi-family v2 producer must reject reserved
task files and use an authenticated identity-only exclusion manifest. A new
untouched confirmation set is required for eventual independent confirmation.

The first profile permits at most288 prompt tokens and128 response tokens
including actual template EOS, with416 total positions. Reject oversize examples;
do not truncate responses or append an invented EOS. It does not qualify the
application's 2,048/6,300-token inputs. The admitted fresh response DEV is separate
from the historical MC DEV; both are explicitly selection data after inspection.

Use exact full-vocabulary forward KL, temperature1, at every response token
including EOS. No CE mixture, hidden-feature auxiliary loss or rollout loss in
this first arm. Original teacher normalized hidden states and native cut20
residuals are stored as lossless BF16 bits and read through authenticated mmap
caches. The original tied readout reconstructs complete vocabulary distributions
in token chunks. Generated contexts must execute their actual native prefix;
teacher-forced cached prefixes may never substitute during generation.

Train one deterministic pass over a prospectively shuffled mixture, stopping at
the first complete accumulation window reaching100,000 supervised tokens. The
exact resulting count is frozen in the machine plan before execution. Use BF16
weights/gradients, FP32 CPU Adam master weights and moments, LR5e-6, clipping1,
weight decay0, microbatch1 and accumulation up to16 complete examples. Checkpoint
at initializer and first windows reaching25k/50k/75k/100k response tokens. Do not
extend epochs in response to quality. Earlier equally eligible checkpoints win.

## Execution gates

Before scientific updates, pass CPU mask/shift/EOS, source-group, cache identity,
dense-versus-chunked gradient, save/reload and actual small-Qwen lifecycle tests.
Run each real-model pilot under a reservation; check finite gradients and FP32
updates in every intended block, actual BF16 storage changes in every block,
fixed geometry/readout, unchanged teacher and saved-state replay. Norm BF16
storage change is diagnostic when an FP32 update is below its quantization step.
Require same-checkpoint full-vocabulary inference parity on a prompt plus two
generated prefixes using the actual unpacked Torch training/inference path.
Pretraining systems probes found that the historical packed executor failed
the full-logit tolerance for both initializers, although all three greedy
decisions per model matched. Maximum absolute differences were0.21875 for
progressive16 and about1.73 for all36. This remains a failed packed-executor
qualification. Independently cloned unpacked HF execution is the additional
control for the actual recovery path; keep the same tolerance
`0.03125 + abs(reference)/128` and require every vocabulary row and greedy
decision to pass. No packed speed/deployment claim follows from an unpacked pass.
A pilot is discarded; scientific training reloads the
authenticated initializer with a fresh optimizer. Concurrent reservations must
cover each actual GPU before either pilot or training begins.

Record CPU optimizer storage, GPU peaks, updates, actual response/input tokens,
phase elapsed time and failure receipts. Use locked/offline Pixi; launch through
`inference-ctl lease run`. Start the progressive run on3060 and all36 on2060 or
2060 SUPER according to the qualified memory pilot. A resource change that keeps
arithmetic/objective fixed is recorded, never hidden as an algorithm change.

## Evaluation and decisions

At initializer and checkpoint boundaries, evaluate all admitted response DEV
tokens for NLL and teacher KL; preserve per-example metrics and denominators.
Run six fixed executable generation canaries plus two deterministically selected
DEV generation examples per admitted source family, greedy up to128
tokens with real EOS. Keep native controls and all raw outputs. This tiny sample
is a collapse/engineering screen, not a population accuracy estimate. Generation
recognizes both EOS IDs from the official model configuration,151645 and151643;
reference-response training retains the actual template EOS,151645.

A separate configured model judges blinded paired outputs for task success,
coherence, source support, numbers/entities/dates and omissions. Preserve route
and returned model identity, rubric, exact request, settings, raw verdict and
failures. Missing/timeout judgments fail eligibility. Numerical/repetition
heuristics are diagnostics, not semantic-summary accuracy. Review a sample of
raw judgments before claiming quality.

Also execute the selected MBPP tests in a bounded bubblewrap sandbox without
network or home/credential access, and score GSM8K final numerical answers.
Require at least50% student success and no more than five percentage points
below native on each of these small paired screens. Missing sandbox/test
evidence fails eligibility while preserving the checkpoint as diagnostic.
Other full-answer exact-match measures are diagnostics; never apply exact
wording as semantic-summary accuracy. These are small selected DEV checks,
not complete official benchmark scores.

Generation eligibility requires at least50% student success and no more than
five percentage points below the paired native control on both the executable
canaries and semantic task/generation screens. Also require complete-response
NLL at most1.10 times native NLL and original four-family MC DEV within five
points overall and ten per family. No metric rescues a failed gate. Among
eligible checkpoints choose higher semantic task success, then lower response
NLL, then earlier update. Preserve the initializer and diagnostic best even if
none qualifies. A failed candidate remains `quality_gate_failed`; an eligible
one is at most `exploratory_quality_pass`, never confirmed or deployment-ready.

Do not automatically advance to1M/10M tokens. If useful generation improves but
does not qualify, the next rung needs a recorded prospective decision. If it
does not improve, diagnose the objective, corpus, inference implementation and
fixed representation before adding epochs. The independently runnable short
student-rollout arm follows dense recovery under a separate frozen budget and
teacher-prefix scoring contract; this initial dense arm does not pretend to
have executed rollouts. Independent TEST confirmation and exported speed/size
qualification occur only after candidate selection is frozen.

Admitted corpus: `data/experiments/jspace_3b_response_data/20261116_v3`,
fingerprint `43a2aed78755d0a467f5f04b9d3c86cb4ab11db0463d478af28b9ab122141297`:
4,441 TRAIN cases with105,089 response/EOS tokens,128 DEV cases with4,875 such
tokens. The first balanced pass uses3,648 distinct TRAIN cases and100,204
supervised tokens over228 updates. Checkpoints occur at updates46/90/152/228,
crossing25k/50k/75k/100k without truncating a response. Artifacts use new dated
`data/experiments/jspace_3b_response_recovery/` roots. Preserve per-arm plans,
source archives, exact checkpoint provenance, pilot/scientific separation and
NAL-compatible status/results. Completed source/test/report records are committed
without large tensor artifacts.
