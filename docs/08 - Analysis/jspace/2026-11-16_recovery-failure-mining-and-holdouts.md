# Recovery training, failure cases and honest evaluation

Research review, 2026-11-16. The user prioritizes usable recovery of the current
three-block Qwen2.5-3B compressed model. The proposed all36 follow-up remains on
hold. This note records recommendations; it does not alter the active frozen
training plan or claim that student-rollout recovery has run.

## Holdouts can be repurposed, but their evidentiary role changes

TRAIN fits parameters and data-dependent geometry. DEV supports checkpoint and
recipe selection. TEST is reserved for final confirmation. Exposed failures can
be deliberately moved into a correction-training pool, but their later scores
become regression tests on known examples. Source documents, conversations,
paraphrases and other derivatives must remain in the same group. A replacement
test must be independent and fixed before further adaptation, not selected for
favorable outcomes. This follows the separation described in
[scikit-learn's evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html)
and its [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Our current128 DEV,256 MC and16 generated-response cases are exposed development
evidence. The historical512-case access incident also requires a new sealed
confirmation set. One answer changes a16-case rate by6.25 percentage points;
that panel can detect collapse but cannot establish the five-point general
quality allowance. Keep known-case regression, new instances of known failure
types, and transfer to withheld task templates as distinct measurements.

## What the primary sources support

| Evidence | Relevant finding and limitation |
| --- | --- |
| [NVIDIA Model Optimizer pruning guidance](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md) | Structured compression is followed by distillation and, when needed, instruction/preference post-training. Its broad-model recommendation is80–100B tokens with4096/8192-token sequences; this is a large-model recipe, not a minimum requirement for our three-block repair. |
| [Nemotron Nano2 technical report, §4.3](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf) | Uses forward-KL logit recovery; its reasoning-model experiments benefit from a mixture of post-training and pretraining data. Later targeted behavior training remains distinct from initial recovery. |
| [GKD, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf) | Supervises student-generated prefixes with teacher next-token distributions and permits mixing fixed examples with student rollouts. Its evidence includes summarization, reasoning and instruction distillation. It does not validate our fitted geometry. |
| [ShortOPD v2, August2026 preprint](https://arxiv.org/pdf/2607.13124v2) | Directly studies generation collapse after Qwen pruning. Adaptive short rollouts reduce repetitive-tail expenditure. Its main run uses about250M rollout tokens; its1024–2048-token horizons are much larger than our proposed32–64. Table14 also shows that better generation need not maximize MC scores. |
| [MiniLLM, ICLR2024](https://arxiv.org/html/2306.08543v4) | Naive reverse-KL optimization can reward short, repetitive or meaningless outputs; sampling and length controls matter. Changing divergence alone is not a reliable cure. |

There is no universal recovery budget or loss mixture. Our105,089 supervised
response/EOS tokens and690,623 processed input tokens constitute an exploratory
local repair budget. Comparisons to all-token pretraining budgets must preserve
these different denominators and compression scopes.

## Practical recommendation for our next experiment

First finish and grade the current three-block pass. Preserve its early
checkpoints. Use a generation-selected checkpoint for a prospective comparison
with the same frozen teacher, geometry, deployment representation and prompt
population. Do not change the active run retrospectively.

Compare equal-supervision arms: ordinary teacher-forced continuation versus a
mixture of clean-response replay and teacher supervision of student-generated
histories. Report actual GPU time as well: these arms will not have equal
runtime. New generated histories require fresh teacher computation and cannot
reuse targets cached for reference answers.

A bounded engineering proposal is256 TRAIN prompts at a32-token cap, followed
by256 fresh TRAIN prompts at64 only if useful completion length warrants it.
This permits at most24,576 rollout positions. Matching that with clean replay
gives at most49,152 supervised positions; use complete-example accounting if
the replay budget crosses a boundary. The ratio and budget are our proposed
pilot settings, not a published optimum or a replication of ShortOPD. Refresh
rollouts between update rounds and identify stale trajectories honestly.

Bad sampled tokens supply contexts, not correct labels. Teacher distributions
provide the training target. Separately collect verified clean responses for
wrong arithmetic, changed numbers/entities, invalid JSON, failing code,
omissions and unsupported summary claims. Use executable/numerical checks
where possible and semantic support checks for summaries. Include genuine
response-ending tokens; a token-cap truncation must not manufacture an EOS.
Teacher continuations can themselves endorse a loop or an incorrect premise,
so check corrections and retain ordinary examples rather than training only on
pathological tails. Preference or unlikelihood training is a later controlled
option, not a default replacement for this comparison.

Select by correct completed answers, supported content, numerical fidelity,
instruction compliance, executable results and natural termination. Continue
tracking broad benchmark retention and DEV loss. A successful small correction
pilot should earn a larger fresh-data budget; lower KL alone should not trigger
automatic repeated epochs. Application inputs of2048/6300 tokens remain outside
the current416-token validated envelope and need separate validation.

The adopted [NAL distillation standard](../../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md)
already specifies a student-rollout stage. The current Qwen run has performed
the dense reference-prefix stage; that later stage is still outstanding.
