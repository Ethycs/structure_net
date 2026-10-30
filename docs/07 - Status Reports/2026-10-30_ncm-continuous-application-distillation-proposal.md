# NCM training from deployed summarizer traffic

Recorded October 30, 2026, America/Los_Angeles, at the user's request to
“log it.” Status: **proposed next experiment; not implemented or running**.
This records the design and does not restart the stopped training or grading.

Use the deployed summarizer as a teacher: continuously save its real input/output
pairs, then train an NCM head asynchronously on those examples. The purpose is
to learn the application's actual conditional task and input distribution.
Our LM1B initialization was pretrained for unconditional generation; subsequent
summary adaptation has not established usable quality. See the
[pretrained-weight audit](../08%20-%20Analysis/2026-10-30_ncm-pretrained-capabilities.md)
and [failure review](../08%20-%20Analysis/2026-10-30_ncm-failure-literature-and-code-review.md).

Proposed data flow:

```text
source + exact summarization request -> deployed teacher -> production summary
                 |                                              |
                 +--------- durable input/output log <-----------+
                                      |
                         training partition + replay buffer
                                      |
                    frozen source features + encoded target
                                      |
                        asynchronous NCM head updates
                                      |
                    held-out semantic and speed evaluation
                                      |
                         qualified candidate checkpoint
```

**Capture contract.** Preserve the complete request and available source,
teacher's raw response, final stored summary, task type (raw-fragment summary
or hierarchy merge), timestamps, source/conversation IDs, batch and fragment
alignment, teacher/model/prompt/tokenizer versions, and available quality
feedback or corrections. Keep original text pairs so retraining is possible
after changing the codec or feature extractor. A successful service response
alone is not a trustworthy quality label. Retain quality status explicitly;
do not automatically treat unverified student outputs as teacher targets.

Current application limits to preserve: 128 stored tokens; inputs below 192
tokens bypass summarization; fragments up to 2,048 tokens; up to eight fragments
within a 6,300-token raw prompt; generation caps of 768 tokens for raw batches
and 512 for hierarchy merges; exact extractor target of 32 words per atom.
Log bypass, extraction, raw-summary and merge behavior distinctly. Preserve
per-fragment targets instead of confusing an entire batch response with one
128-token summary. The application's storage tokenizer and the NCM codec's
native frame have different capacity limits; never silently truncate targets
to fit the latter.

**Training contract.** Keep the serving model frozen and update the NCM branch
in a background worker. An attached head consumes detached source-only states
from the body/layer that will actually be available before generation. Teacher
answer states must not leak into its inputs. An API-only teacher can provide
labels while a separate local frozen body supplies the student's features.
Bind saved features to the exact body, tokenizer, prompt and layer versions.

First qualify codec reconstruction on the application's summaries, especially
names and numbers. Freeze and version the codec during a head-training stage;
re-encode stored targets when it changes. Mix fresh examples with older replay
examples. Use bounded disk batches and periodic checkpoints rather than placing
optimizer steps on the production request path. Source-conditioned learning
and free-running generation must be evaluated explicitly; merely adding more
pairs does not repair the rollout failures already observed.

**Evaluation and promotion.** Assign whole source/conversation groups to stable
training or evaluation partitions before updates. Prevent duplicate sources,
overlapping hierarchy branches and later revisions from crossing those
partitions. Keep the evaluation groups out of the replay buffer. Compare
complete, freely generated summaries against source facts, accepting valid
paraphrases. Evaluate faithfulness, salience, readability, numerical accuracy
and the storage limit; teacher imitation and code-token agreement are diagnostic
metrics, not application accuracy.

Retain the application goal of 95% complete-summary semantic accuracy and the
speed objective of 2× tokens/s at a matched input/output range. Define sample
size, uncertainty, judge qualification, fixed baselines and stop rules before
running the study. Measure the complete inference path, including the frozen
body, and report ongoing training/storage costs separately. Serve evaluated
checkpoint snapshots, not a head whose weights mutate during a request.

**Existing components to reuse:**

- [Source-conditioned head](../../experiments/structure_net/ncm_lm1b_source_attention.py)
  and [training path with learned-root gradients](../../experiments/structure_net/ncm_trainable_example_batching.py).
- [Packed source-feature storage](../../experiments/structure_net/ncm_packed_feature_store.py).
- [Disk-batched source/target/code joins](../../experiments/structure_net/ncm_recovered_training_stream.py).

These are reusable research components, not a completed production integration.
The deployed capture entry point, teacher/body versions, data-admission rule,
codec choice, update budget, replay mixture and evaluation schedule remain
implementation decisions. No continuous worker or live traffic capture has
been enabled by this note.

[Online Speculative Decoding](https://proceedings.mlr.press/v235/liu24y.html)
provides precedent for adapting smaller models using serving traffic.
[On-Policy Distillation](https://arxiv.org/abs/2306.13649) studies feedback on
student-generated sequences. Neither establishes that this NCM adaptation will
succeed; NCM code-space supervision requires its own implementation and tests.
