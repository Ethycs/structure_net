# WORD_GAMES broader coding: HumanEval labels

Date: 2026-11-12. Prospective exploratory study:
`word-games-humaneval-labels-v1`. No primary model outcomes inspected.

Question: how often does the stronger-controller/Qwen execution pipeline solve
public function-coding problems, and does consistent interface relabeling reduce
that rate? This broadens task coverage beyond the eight reservation exercises.
It does not test repository maintenance, autonomous metaphor choice, information
secrecy, or the production bridge HTTP service.

Use 32 of the 164 problems from the official OpenAI HumanEval repository, revision
`b9453e8f1865df0a0057a1383767caa400acfbcf`. Select without replacement using Python
Random(731113), sort the selected numeric IDs, and retain all selected tasks:
5, 6, 21, 23, 26, 31, 32, 34, 35, 41, 45, 46, 52, 54, 56, 59, 62, 63, 65, 66,
72, 78, 79, 95, 113, 124, 133, 139, 141, 152, 159, 160.
Data and license are preserved under
`data/experiments/word_games/20261112_humaneval_data_v1/`.

Terra interprets each public prompt, including its public examples and any
provided helper code, into a complete instruction. It must preserve requirements
and provide no implementation. The initial instruction is shared between arms.
The direct arm implements the original interface. The relabeled arm uses Qwen to
translate the instruction into a deterministic one-to-one function/parameter
codebook: entry point becomes `bridge_operation`, parameters become `parcel_N`.
Actual literals, mathematical operations and return values stay unchanged.
Relabeling changes code identifiers, not arbitrary natural-language nouns.
This is a different and narrower mapping than a full domain metaphor.

Preserve imports and supplied helpers in each initial solution file. Strip the
entry-point docstring and replace its body with a stub. For the relabeled arm,
AST-based identifier renaming changes names and parameter bindings, preserving
literal constants and annotations. Both actors get only their instruction,
interface and supplied scaffold, never canonical solutions or hidden test code.
The target actor receives no inverse codebook or original task prompt. Host-side
test invocation adapts keyword names but leaves official assertions unchanged.
Before inference, all 64 original/renamed canonical solutions must pass those
official tests in the same sandbox used for grading. Wrong-solution, timeout and
completion-gating controls must fail as intended.

Use existing pinned Terra and Qwen routes, OpenCode 1.19.5, bubblewrap, 8192 context,
1024 actor output tokens, 12 agent steps, temperature zero, no compaction. Keep
translation output at 768 tokens and controller output at 1024 tokens. Alternate
arm order, use fresh sessions/workspaces, and permit at most two typed requests
for a simpler controller instruction. No grading feedback or answer repair.
Record every call, clarification, solution and failure. Do not retry failed
answers or replace sampled tasks. Stop for infrastructure/route/fingerprint
failure; retain unattempted tasks explicitly, without treating them as measured
model failures. Model context, truncation, protocol and clarification failures
count against task completion and continue to the next task.

Primary: proportion of the 32 tasks with all official assertions passing and a
normal `stop` from the final actor turn. Secondary: artifact-only passes, paired
direct-only/relabel-only successes, observed rate difference, descriptive Wilson
95% intervals, per-task latency, failure categories and clarification counts.
One generation per task/arm, so this is observed pipeline pass@1 on a sampled
benchmark, not a leaderboard-compatible bare-model score. Report all failures;
do not use individual assertion counts as independent samples or claim 95%
general coding accuracy. Public benchmark training contamination is unknown.
Thirty-two tasks cannot establish population-level equivalence. Classification:
`sampled_public_coding_evaluated`, with broad confirmation false regardless of
score. Equal paired successes mean observed parity on this sample only.

Fresh intended roots: `20261112_humaneval_labels_v1` and a separate audit root
under `data/experiments/word_games/`. Freeze data, source, plan and configuration
hashes before inference. Independently reconstruct grades and rerun captured
programs in network-isolated sandboxes. Preserve earlier results unchanged.
Run via Pixi from the repository root; execution workspaces belong in `.pixi/`.
No GPU/service changes, training, or DataAnnotation material.
