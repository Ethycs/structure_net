# Evidence teacher: recover one leading JSON block

2026-10-24. Hypothesis `ncm-evidence-first-fresh64-teacher-generation-v3`.
Preserve both failed implementations and their actual exit1receipts. V2correctly
refused to skip original first response0: one Sonnet response contained a valid
leading JSON code block followed by explanatory prose. The gateway also reported
3744completion tokens despite the512requested cap. Do not claim the request cap
is verified actual gateway usage; retain all raw usage/response fields.

This third wrapper allows exactly one leading JSON code block and trailing prose
that contains no braces or further code blocks. Parse the sole block and keep
the original summary/support strings without edits. Reject multiple candidate
objects or blocks. No semantic selection, request reroll or prompt change.
Before any new calls, verify all four original pilot attempt0responses parse
under this formatting-only rule. Reuse those exact first responses, never later
original attempts. Both recovered pilot candidates stay in the same fixed64.

All source selection, inputs, source-line IDs, models, generation/review prompts,
request caps, capacities and prospective semantic qualification requirements are
unchanged fromv1/v2. Freeze and independently replay this wrapper and all four
original first response bytes as well as all64final first candidates. Keep
failure denominators. No canonical semantic cache change or training admission.
