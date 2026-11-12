# HumanEval failure mechanism diagnosis

2026-11-12. Exploratory follow-up: `word-games-humaneval-failure-diagnosis-v1`.
The 30/32 official scores and the extra /160 associativity failure are already
known. This is selected-case diagnosis, never a replacement accuracy estimate.

Read and hash-check the frozen controller instructions, translations, tool traces
and programs for HumanEval/32, /113 and /160 in `20261112_humaneval_labels_v1`.
Keep all original evidence unchanged. All work runs through Pixi from the repo
root; fresh execution workspaces go in `.pixi/`, evidence in new `data/` roots.

Offline interventions on COPIES of generated programs:

- /32 native: make the existing bracketing loop actually expand its bounds.
- /32 relabeled: connect the required entry point to its already-written
  `find_zero` function, changing no arithmetic.
- /113 relabeled: replace every lowercase `i` in the exact literal template;
  leave odd-digit counting unchanged.
- /160 native: correct operator-versus-operand indexes and the exponent-loop
  cursor; preserve the existing three-pass algorithm.
- /160 relabeled: traverse exponentiation operators from right to left, keeping
  the lower-precedence passes unchanged.

Run original, corrected and canonical controls on the official suites and fresh,
deterministic targeted cases (linear/cubic roots, odd-digit counts, arithmetic
operator combinations and exponentiation chains). Record concrete expected and
observed values where execution completes. Corrections are researcher-authored,
test-informed diagnostic interventions; passing them is not model learning or
evidence that the deployed system repairs itself.

Live handoff isolation: for /32 and /113, run two fresh actor invocations for each
of original and corrected translated instructions (eight total), alternating
condition order by repeat. /32 correction changes only the function/parameter
names to the declared target interface. /113 restores only the missing suffix of
the already available controller instruction, after deterministic name mapping.
No generated implementation, hidden tests, failure messages or diagnostic fixes
go to the actor. Use the same scaffold, prompt, permissions, model route, 8K
context, 1024 output-token cap and 12-step budget as the original study. The
controller and translator outputs are frozen for this intervention; this is not
a new end-to-end controller benchmark. No retries or answer repair.

Report every live result and distinguish normal completion, official test
success and additional-case success. Repeats are fresh sessions at temperature
zero, not independent model seeds. Stop live work for transport/route failure,
retain partial results, and do not pool a repaired campaign into the original.
Offline cases are generated before live results. Reconstruct scores from raw
grades and replay final programs. Classify `selected_failure_mechanisms_tested`,
with broad accuracy and autonomous-repair confirmation false.

Specific uncertainties: whether the incomplete /113 example caused that output
error cannot be established from the original trace alone; the replay tests its
local effect. A normal API stop does not establish semantic completeness. This
study does not diagnose sampler internals or prove that grammar caused the early
end. No deployment, model training, service restart or DataAnnotation material.
