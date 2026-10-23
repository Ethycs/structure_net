# NCM training-regime recovery

Date: 2026-10-22. Hypothesis `ncm-paper-regime-recovery-v1`. EXPLORATORY,
underpowered, one model seed. Earlier 0/64 results and failed 32-example
competence gates remain failures. This is a new study, not their continuation.

Question: can staged adaptation, substantially better paired data, and a
correctly packed frozen-Llama context recover useful conditional generation?
The user's eventual target remains >=95% complete-summary accuracy, followed
by a measured throughput comparison. This study cannot guarantee that target.

Primary source: https://arxiv.org/html/2610.14162v1, sections 3.3–3.4, C, F.2–F.3.
The paper trains the tokenizer first, freezes it, pretrains the generator on
unpaired text, then trains source–target conditioning. Nested dropout applies
to tokenizer training; conditioning corruption applies to generator inputs,
with clean target codes. A frozen T5 encoder is an evaluated alternative.
The released XSum checkpoint uses nine levels although the paper table lists
eight. Preserve the released nine-level schedule and disclose this difference.
This resource-scaled adaptation is not a reproduction of 350,000 updates at
batch 512, nor evidence that NCM already achieves 95% semantic accuracy.

## Data stage fixed before generation

Use the existing 2,048 training passages and their cached frozen Llama body
features. Verify their original campaign hashes. Ignore their old summary
labels. Generate new labels using `claude_code/claude-sonnet-4-6`, temperature
zero, 24-word request, four independent passages per JSON request. Keep every
raw response, including unusable labels. No semantic-failure retries. At most
three attempts for transport or malformed JSON. Labels exceeding 47 GPT-2
tokens are excluded without truncation. Require at least 1,792 eligible labels.
The historical 32-case training attempt is not reused as the new initialization.

Select 64 development passages and seal 256 final-test passages, in deterministic
ID-hash order, from the original test partition. One source group per passage;
exclude groups used by all earlier 64/256 quality cohorts and calibration.
Training, development and final test are disjoint by group, session and source
hash. Source eligibility: existing native constraints and <=512 rendered Llama
prompt tokens, without truncation. No selection using output quality. The
test is opened only after an explicitly declared development competence gate.
The 48-token target canvas remains a subset of the application's <=128-token
budget; full 2,048-token fragments and batched hierarchy merges remain outside
this first recovery study.

Before any outgoing source payload, verify the full public LongMemEval file
against its published SHA and reconstruct exact source spans. Send only those
public passages and generated labels; no saved private labels or internal
metadata. A serial request and eight-worker concurrency pilot precede the
teacher batch. Pilot labels are systems-only and not pooled into training.
Audit a fixed 32-label sample with the independently qualified secondary judge;
require >=31/32 valid passes before conditional quality-training claims. Preserve
valid failures; retry only transport/schema problems. Do not weaken the rubric.

## Training stages

Before implementation launch, record exact numerical training configuration in
an addendum. Start from the released XSum codec and generator, avoiding the
earlier sequence of low-quality fine-tunes. Check reconstruction on new labels
before deciding whether tokenizer retraining is needed; a pretrained tokenizer
may satisfy the first paper stage. Require >=98% non-PAD reconstruction and
source-grounded oracle-decoding competence. If it fails, stop the conditional
quality path and separately implement genuine tokenizer adaptation with encoder,
EMA codebooks, nested dropout, commitment and quantization objectives. Merely
adapting a decoder on frozen codes is not full tokenizer training.

Freeze the accepted codec before encoding generator labels. Adapt the generator
on unpaired public training text, then conditional pairs. Use a null prefix for
unpaired training. Use same-level uniform conditioning corruption and clean
targets, BF16 autocast, FP32 code assignment and optimizer parameters, separate
seed streams, clipping and AdamW. Pack valid frozen Llama states to the beginning
of the prefix before the bridge, matching NCM's right-padding convention.
Warm up the bridge with the generator frozen, then train bridge and generator;
the Llama body stays frozen. Include a native-prefix NCM control to distinguish
data/codec failure from failure of the Llama interface.

Require CPU contract tests, a real CUDA update/save/reload pilot, frozen-codec
identity, nonzero adapter/coarse-query gradients and representative memory before
full training. One worker on physical GPU 0; preserve resident services. Stop on
hash mismatch, nonfinite values, OOM, split leakage or failed paired reload.
Use new dated artifact roots under `data/experiments/ncm_paper_regime/` and
content-hash/preserve all producing sources and inputs. Pilot and full stage
identities must differ. Completed evidence is append-only.

Development accuracy uses the unchanged whole-summary faithfulness, salience,
readability and <=128-token rubric. Token reconstruction, CE, code accuracy,
ROUGE and memorization are diagnostics. Final success requires >=95% of all
sealed test passages to pass independently under both qualified judges, complete
valid grading, and oracle/teacher positive controls. Report Wilson intervals
and judge disagreement; no human-evaluation or production-readiness claim.
Failures motivate new dated studies, not retries until a favorable seed appears.
No speed claim before quality. Save aggregate, report and conservative registry
read-back for each stabilized result.
