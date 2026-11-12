# WORD_GAMES stronger-controller coding rerun

Date: 2026-11-12. Prospective exploratory design.
Study: `word-games-strong-controller-coding-v1`.

Run the same eight synthetic coding tasks and four exact hidden cases per task
from the completed 2026-11-11 pilot. These are a known regression set, not new
held-out tasks. Reuse that pilot's per-task alias dictionaries so alias-dependent
ordering and masking do not change between runs. No DataAnnotation data.

The forward architecture is a stronger controller interpreting the task, Qwen
translating the instruction, then Qwen executing it in stock bubblewrapped
OpenCode. The registered `codex_sdk/gpt-5.6-terra` route passed an identity-checked
preflight. Pin its observed deployment ID and the existing Qwen deployment ID.
Report gateway model aliases; these are not independent weight attestations.

For each task, Terra receives the original requirement and input schema/types,
without concrete examples, hidden cases, expected answers or reference code.
It produces one short implementation instruction shared by both arms. Native
Qwen executes this directly with the original task data. In the translated arm,
Qwen first translates it through the existing mapping and then executes with the
existing Privacy+Mapping view. Keep the full existing manuals, exact transforms,
8K context, 1,024 executor output tokens, 12 steps per invocation, disabled
compaction and actual no-network execution grader. Do not quietly fix known
wire-name collisions or introduce missing text/order predicates during this run.

Both arms can ask for simpler instructions using the typed pause contract. At
most two controller revisions per arm, counting translator and executor requests
together. Forward only fixed reason codes and the requesting role to Terra.
Terra receives its previous source instruction and original requirement, not
actor prose, source data, test outputs or the inverse alias table. Every revision
in the translated arm passes through Qwen translation before execution. Resume
the same native OpenCode session on an executor request. Requests that remain
unresolved are failures, not completed answers. This run uses normal executor
self-requesting behavior; it does not insert a mandatory read-only review turn.

Alternate arm order. Never send grading feedback to either model, repair answers,
or pool a stopped run into a new run. Context overflow, timeout, malformed model
output and exhausted clarification budgets count as task failures. Transport and
route-identity failures stop the campaign and retain its partial evidence.
The shared initial controller instruction must validate before a pair can start;
an invalid initial instruction also stops the campaign rather than inventing one.

Primary outcome is complete tasks passed out of eight; secondary outcomes are
hidden tests passed out of 32, artifact-only grades, clarification counts,
resource failures and per-task latency. Charge the shared initial controller
latency to both arms when estimating individual task latency. Preserve raw API
responses and OpenCode events. The main comparison is matched native versus
translated performance in this run. Comparison with the old 8/8 and 3/8 pilot is
descriptive: controller instructions and pause handling are new interventions.
This does not establish general coding accuracy, information secrecy, an isolated
causal benefit of simplification, or a production HTTP bridge deployment.

Freeze source and input hashes before calls. Use a fresh artifact root under
`data/experiments/word_games`; commit the producer, tests, design and result report.
