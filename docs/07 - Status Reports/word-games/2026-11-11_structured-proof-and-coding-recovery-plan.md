# WORD_GAMES structured proof and coding recovery

Date: 2026-11-11. Status: prospective exploratory plan.
Study ID: `word-games-structured-proof-coding-recovery-v1`.
This is not a rerun or rescue of a prior accuracy campaign. Use only the supplied
synthetic reservation fixture and newly authored coding exercises; no
DataAnnotation material, private application data, or subscription model route.

## Structured claim and bounds

Extract the actual mapping from the previously verified synthetic service session
`190b56341e34b99b025afd6535e07b03`, reading SQLite without changing the service.
Freeze the table and hash the Engine, Mapping, Bridge, context exporter and producer.
Check each of key_map, enum_map and id_prefixes for totality and injectivity, and
check prefix ambiguity. Do not confuse these finite tables with a bijection over
arbitrary JSON: the existing codec drops legacy target_id and preserves free text.
Record concrete counterexamples to an unrestricted claim rather than hiding them.

The source transition is the real Engine.apply. The target transition is the real
Bridge.apply on the encoded action list with the same host-issued context. There
is no independent target-domain Engine. The precise equality is
`T_phi(phi(s), phi(a); c) = phi(E(s, a; c))`, where T_phi is the deployed adapter,
and the projection drops legacy target_id. Also compare the complete source states,
results, receipts and rejected-state rollback before that projection.

Enumerate 21 valid starting placements of bronze and atlas over no placement or
the four cells north/N1, north/N2, archive/A1, archive/A2, excluding collisions.
Other fixture records remain fixed. Use eight context regimes: full, north-only,
bronze-only, reserve-only, viewer, north-curator, stale revision, stale rule.
Freeze a finite event alphabet in the producer, including all five operations,
missing/mismatched approvals, invalid dates, denied destinations, duplicate IDs,
empty/protected-text annotations and creation. Exhaust every action word of length
0, 1 and 2 as an atomic transaction in every state/context combination; repeat
each transaction ID once to check replay. This is exhaustive over that declared
finite domain, not over all strings, dates, states or transactions up to the
engine's 20-operation limit. A deliberately corrupted decoder must be detected.

Success requires zero transition, state, replay and rollback counterexamples over
the entire declared domain, nonzero accepted and rejected cases, all operations,
and the positive fault-detection control. A failure is reported without silently
patching the measured production codec. Also give the algebraic reason the adapter
commutes when the typed codec has a left inverse; do not claim an independent
implementation or natural-language semantic proof.

## Coding and information-recovery pilot

Use newly authored small Python coding tasks over the same public synthetic
reservation data. Freeze eight tasks and hidden input/output tests before inference.
The native baseline reads the source specification/data; the translated arm reads
only a Qwen-translated specification and the existing codec's mapped, privacy-shielded
data. Both use stock OpenCode 1.19.5 with the existing Qwen3.8-35B-A3B route, identical
budgets and fresh bubblewrapped workspaces. Required answer: solution.py with
`solve(data)`. Execute answers in a separate no-network bubblewrap with bounded
time and hidden test inputs; grade actual returned values and input preservation.
Decode translated outputs through the same inverse codec and alias table.

This is a coding adapter evaluation of the deployed codec/privacy components,
not the reservation HTTP API executing arbitrary code and not SWE-bench parity.
Do not put expected solutions in actor prompts. Keep failed model outputs and
score them without repair/retry. Use one development-only toy startup check if
needed; do not tune on any of the eight measured tasks.

Measure exact task pass counts, hidden-test pass counts, source literal recovery,
source-domain guessing, and per-key inversion of genuinely renamed fields.
Downstream recovery calls get only the receiver packet and the public source
schema vocabulary, never the table, alias inverse, source files, hidden tests or
another arm's history. Include no-view controls and raw-view positive controls.
Distinguish intended recovery of task-relevant mechanics from recovery of original
names or irrelevant synthetic private canaries. Also scan raw Mapping output
versus the deployed Privacy+Mapping composition, and run an oracle inverse control.
Report denominators and paired outcomes; eight tasks with one model are not an
accuracy estimate for general repositories or a proof of information secrecy.

All calls use the existing local Qwen service via LiteLLM/Triton. Transport or
route failures stop the run and remain in the record; no cloud fallback. Results,
prompts, model events, tables, grading and producer hashes go into fresh immutable
directories under data/experiments/word_games. Source, tests, plan, report and
lightweight provenance are committed. Preserve unrelated worktree changes.

Pre-inference clarification: domain/name/canary recovery is a **blind** call,
without source-schema vocabulary, so that giving it words such as "galleries"
does not reveal the answer. A separate strong, schema-aware call attacks only
genuinely renamed keys. No-view and raw-source controls are separate fresh calls.
The extracted real table renames only 8/52 keys; report the 44 unchanged keys
separately. Task definitions and four hidden cases per task are frozen in
`coding_recovery_tasks.py` before any task inference. Exact lexical ordering and
text predicates remain part of the requested functions even if opaque aliases
lose that information; do not silently relax those tests to improve parity.
