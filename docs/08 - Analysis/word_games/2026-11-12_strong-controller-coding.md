# WORD_GAMES coding rerun with a stronger controller

Date: 2026-11-12. Status: completed and independently audited.
Study: `word-games-strong-controller-coding-v1`.

**Coding quality did not improve in this rerun.** The direct arm produced working
programs for 8/8 tasks (100%); the translated arm produced working programs for
2/8 (25%). The prior pilot's translated score was 3/8 (37.5%). A completion audit
adds an important qualification: one of the two passing translated programs had
a truncated final explanation. Requiring both correct code and a normally ended
model turn gives **1/8 translated tasks (12.5%)**, versus 8/8 direct.

These are eight previously inspected synthetic regression tasks, not a general
coding benchmark. No DataAnnotation material was used. The old-pilot comparison
is descriptive; the new controller and clarification protocol are simultaneous
changes, so this is not an isolated estimate of either one's causal benefit.

## Architecture actually exercised

The automatic evaluation path was **Terra interprets → Qwen translates → Qwen
executes**. The direct control bypassed the translation pass and used the same
Terra initial instruction with the same Qwen executor. Terra used the registered
`codex_sdk/gpt-5.6-terra` route, pinned to deployment
`5d42c0e9-a52d-43b0-9da9-4f9bbea67599`. Qwen used `qwen3.8-35b-a3b`, pinned for
translation to `8b54306d-a232-4c50-8ca6-9defe9969133`. These are verified gateway
aliases and deployment identities, not an independent inspection of model weights.

Terra received each original requirement and a schema/type summary. It received
no concrete example values, reference implementation, hidden expected answers or
grading feedback. Its initial instruction was shared between arms. In the
translated arm Qwen translated that instruction, then a fresh bubblewrapped
OpenCode 1.19.5 session implemented `solve(data)`. The same session would resume
on executor clarification. Arm order alternated. Both arms retained 8K context,
1,024 output tokens, 12 steps per actor invocation and disabled compaction.

The real simplification loop ran automatically: three valid translator requests
produced three additional Terra calls and Qwen retranslations. There were no
executor-originated typed requests. Both tasks that requested clarification failed
before reaching execution. No human repaired their instructions or answers.
There were 11 controller API calls, 11 translation API calls and 14 actor
invocations (the actor invocations contain their own native model/tool steps).
The production bridge HTTP service was not changed by this standalone evaluation.

## Results

| Task | Direct tests | Translated tests | Translated outcome |
|---|---:|---:|---|
| Free slots | 4/4 | 0/4 | Context overflow; no implementation |
| Earliest reservation deadline | 4/4 | 4/4 | Correct code; final explanation truncated |
| Count by category | 4/4 | 0/4 | Two clarification requests, then invalid mixed verdict |
| Minimum inherited deadline | 4/4 | 0/4 | One clarification request, then invalid mixed verdict |
| Case-insensitive note substring | 4/4 | 1/4 | Opaque text does not preserve the predicate |
| Alphabetical title ordering | 4/4 | 1/4 | Opaque handles do not preserve title order |
| Missing/mismatched approvals | 4/4 | 0/4 | Field-name conflict; generation truncated; stub remained |
| Newest matching creation event | 4/4 | 4/4 | Pass |
| **Total** | **32/32** | **10/32** | **8/8 versus 2/8 working programs** |

The exact grader ran the actual code in a separate network-isolated bubblewrap,
checked decoded results and input preservation on four cases per task, and kept
all failures. The artifact-only counts equal the primary counts in this run.
The previous run had 32/32 direct tests and 14/32 translated tests.

Median task latency was **31.64 seconds direct versus 37.02 seconds translated**,
including failures and charging the shared initial-controller latency to each arm.
The complete campaign took 474.43 seconds. These timings include controller,
translation, actor process and tool overhead; they are not decoding tokens/s.

## What failed and what the audit caught

1. The free-slot request exceeded the context limit: 8,254 requested tokens versus
   8,192 available. This was counted as a failure, not retried with a bigger budget.
2. Category counting and deadline aggregation returned `kind: instruction` with
   a nonempty `needs` array. The decoder schema allowed that combination, while
   host validation rejected it. This is a protocol-design mismatch to fix with
   mutually exclusive schema branches, not evidence that the task is unsolvable.
3. The existing untyped goal rewrite merges field and enum namespaces. It turned
   field `medium` into `aircraft_type`, and `restricted` into `maintenance_window`,
   although the transmitted data fields remained `medium` and `restricted`.
   The approval actor explicitly noticed the missing field, reasoned until a
   `length` finish at the 8K limit, and left `NotImplementedError` in the file.
4. Text matching and title sorting still operate on random opaque strings.
   No stronger controller instruction can make those strings preserve the
   omitted substring or lexical-order relation. This run added no authorized
   text-predicate or ordering service to supply the missing information.
5. The launcher checks CLI exit and explicit error events, but not the final
   model `finish_reason`. Deadline selection produced correct code, then ended
   with `length`; approval checking did the same without producing code. Thus
   the frozen completion rule counted deadline selection as a pass. The audit
   preserves that primary result and separately reports a normal-stop metric:
   **direct 8/8 tasks, 32/32 tests; translated 1/8 tasks, 6/32 tests**.

Clarification currently forwards coarse fixed need codes. A request such as
`resolve_conflict` does not identify the offending translated field to Terra.
The observed failures suggest the next change should provide typed references
to the conflicting clause/field, repair the field/enum rewrite, enforce a
disjoint instruction-or-request response schema, and reduce irrelevant context.
That recommendation is an inference from the traces, not a demonstrated fix.

## Controls, limitations and provenance

The task cases and private alias dictionaries were copied from the old pilot and
hash checked. This keeps encoded data comparable, including alias-dependent
ordering. The runner seeds the complete grading alias dictionary before goal
rewriting. A consequence visible in the approval instruction is that the ordinary
word `absent` is replaced by a payload handle. Thus lexical goal shielding also
depends on which aliases have previously been registered; it must not be described
as a pure field renaming. Original test inputs and expected answers were not
sent to either model during interpretation or clarification.

All outcomes were frozen before independent auditing. The auditor reconstructed
decoded outputs from raw evaluator responses, checked every grade, source and
input hash, and checked the controller/translation route identities. Its first
attempt assumed a clean-exit truncated turn could not have passed code tests;
the assertion failed on deadline selection. That audit attempt is preserved.
The revised audit reports both the original rule and stricter normal-stop scores
without editing the producer's results. Thirty focused tests passed for the
orchestrator, feedback protocol, grader and conservative registry record.

Run: `data/experiments/word_games/20261112_strong_controller_coding_v1/`.
Result SHA-256:
`197f07c257e7b0ec4795aab1b4d46f3f08a79cc38e51459eaa1ccb7c0efe6b64`.
Audit: `data/experiments/word_games/20261112_strong_controller_coding_audit_v2/audit.json`.
The registry keeps `confirmed: false`; no general accuracy or confidentiality
claim is established.

```sh
pixi run -e word-games python -m experiments.word_games.controller_coding --output data/experiments/word_games/NEW_CONTROLLER_CODING --key-env-file .env
pixi run -e word-games python -m experiments.word_games.audit_controller_coding --run data/experiments/word_games/NEW_CONTROLLER_CODING --output data/experiments/word_games/NEW_CONTROLLER_AUDIT
```

Use fresh output roots. The original 2026-11-11 run remains intact; neither failed
cases nor unfinished/model-invalid turns were silently removed from denominators.
