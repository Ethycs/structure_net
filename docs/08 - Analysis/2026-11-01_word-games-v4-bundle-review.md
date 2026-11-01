# Neighborhood Bridge V4: verified bundle and WORD_GAMES integration gaps

Reviewed the supplied `Neighborhood_Bridge_V4.zip` on November 1, 2026.
Archive SHA-256:
`c3c063064d3450cb8a45b78d2c3874bea865a11f4eaebee92145430ad2f5910b`.

V4 contains a substantially richer bridge than the current WORD_GAMES adapter:
model-authored operational instructions, full schema mappings, an executable
symbolic codec, and translated task checkpoints. The active integration still
imports V2 and uses eight model-selected metaphor labels with fixed host
rendering. V4 has not been installed into the active bridge or serving model.

This review made **zero new model calls**. Local tests, retained-output replays,
deterministic disclosure probes and read-only tokenization were run through
Pixi. The original ZIP and active vendor/runtime remain unchanged. Audit commands
were allowed to rewrite derived files only in a separate `.pixi` work copy.

## What V4 adds

| Capability | Current WORD_GAMES | Supplied V4 |
| --- | --- | --- |
| Domain adaptation | Eight metaphor labels | Model-generated target identity, objective, rules and operational manual |
| Symbolic mapping | Fixed action/identity projection | Bijective key, operation-enum and ID-prefix maps with encode/decode |
| Target task solver | Receiver reports a host packet after sender execution | Fresh target-only solver proposes target-schema actions from a translated goal and scoped current state |
| State continuity | Host Engine across events | Same Engine plus encoded/decoded carried task checkpoint and resume checks |
| Recursive procedure | No capsule/self-translation integration | Capsule carries translator and self-translation instructions; externally invoked G0 → G1 → G2 artifact generation |
| Model orchestration | Persistent Triton client and bounded sender tool loop | Prompt/ingest CLI; fresh external/native model calls must be supplied |
| Model-executed code or automatic recursive model calls | Absent | No general REPL or automatic recursive model-call runtime supplied |
| Cross-agent disclosure | Explicit allowlist, stable public aliases, restricted note codebook | Scoped source view with protected literal titles/notes retained |

The V4 source Engine is **byte-identical** to our current vendored V2 Engine.
The difference is the mapping, solver interface, instructions, checkpoint
representation and orchestration around it. Replacing the Engine alone would
not bring in those capabilities.

V4 recursion is bounded declarative self-translation. The fixed Python runtime,
JSON ABI, authority and model infrastructure are not translated. Its bridge
protocol counters/phase are preserved, but the translated lifecycle is not
executed autonomously. Task checkpoint resumption is separately demonstrated.
The supplied harness uses explicit fresh-agent prompt/file handoffs; its own
documentation says this is instruction-level isolation, not an OS sandbox.

## Local verification

- All **196 delivered-manifest file hashes** matched before replay.
- **10/10 V4 unit tests** and **5/5 independent audit-checker tests** passed.
- All **15 frozen core files**, oracle and evaluator hashes remained intact.
- Baseline and descendant saved trajectories each replayed **6/6 events exactly**.
- Independent replay checks found **6/6 operation-or-clarification checks** and
  **6/6 full-state matches** in each trajectory; both end at revision 4.
- G1 checkpoint resume preserved the previous reservation, old note and other
  records while committing exactly the requested new annotation.
- Four generated/control mappings each passed **10/10 codec/protected-literal
  probes**, with invalid-response controls rejecting without mutation.

Manual inspection of the retained clarification outputs agrees with their
stated reasons: H04 identifies an inherited deadline that an attribute exception
cannot waive; H06 asks which of two identically titled records is intended.
This inspection is not a new independent model judge.

The historical failure remains: the first G2 capsule is invalid JSON. Its
separately declared retry passes capsule/mapping checks, but no downstream
six-event G2-retry solver run exists. The instruction-ablated control also
produces a valid mapping; translated instructions' causal benefit is therefore
unproven. The archive records 22 historical fresh-agent calls, but its ledger
does not identify a provider/model version. These results are not established
Qwen results and cannot be compared directly with our different normal-use cases.

## Measured integration constraints

**Disclosure policy must remain outside the codec.** A synthetic marker placed
in the permitted Bronze Heron record's notes survives V4 `Bridge.view()` and
its vocabulary translation. The same marker is absent from the current
WORD_GAMES receiver packet. This follows V4's intentional preservation of literal
payloads; it does not implement our stricter cross-agent note-disclosure policy.
Installing that view unchanged would broaden what the receiving agent sees.

**Fresh mapping/capsule generation exceeds current output limits.** Counts below
use the deployed Qwen tokenizer, excluding chat framing:

| Supplied artifact | Prompt tokens | Saved output tokens | Combined |
| --- | ---: | ---: | ---: |
| G1 self-translation | 5,680 | 6,309 | 11,989 |
| G2 retry self-translation | 6,487 | 6,151 | 12,638 |
| G0 fresh mapping | 2,075 | 2,808 | 4,883 |
| G1 fresh mapping | 2,371 | 3,380 | 5,751 |

Our adapter caps generation at 1,024 tokens and native context at 8,192. Full
capsule regeneration needs a different token budget/context configuration or
an explicitly tested staged design. Mapping generation fits the context in
these examples but exceeds the output cap. Representative saved target-solver
prompts H01/H02 are 3,121/3,329 tokens, so task solving with an already validated
mapping is a smaller initial integration than capsule regeneration. No larger
context GPU-memory or throughput measurement was made.

## Practical consequence

V4 is the appropriate source for the richer mapping-and-checkpoint architecture
that our simplified integration omitted. A faithful WORD_GAMES integration should
preserve the present host disclosure policy, adapt the target view and ID binding
to authorized fields, and use the V4 codec/manual/target-solver path. Generated
capsules should be validated before installation, with model calls connected to
the existing Triton transport under explicit token budgets.

Then repeat the live normal-operation cases and disclosure tests, including the
name-resolution and remaining-occupancy failures. The current speed optimizations
can remain a serving configuration, but their measured throughput and quality
cannot be assumed for these larger prompts or a changed model workflow.

No integration or fresh model evaluation was performed by this review.

## Evidence and reproduction

Evidence: `data/experiments/word_games/20261101_v4_review_v1/`.
`result.json` links hashes for extraction, delivered integrity, unit/audit logs,
replay summary, disclosure comparison and token counts. The extracted work copy
is `.pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/`; preserve both sibling
`bridge_v4` and `bridge_v4_audit` directories. The archive is the unchanged
source of truth, because audit commands rewrite derived work-copy artifacts.

Review result SHA-256:
`efc1a0368198a065fd96a11a2a1e3e63b19bd87fb1c743290f802f55a6ad9305`.

From repository root, the verification commands are:

```bash
pixi run python -m unittest discover -s .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4/tests -v
pixi run python -m unittest discover -s .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4_audit -p test_audit.py -v
pixi run python .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4/src/run.py verify-freeze
pixi run python .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4/src/run.py replay baseline
pixi run python .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4/src/run.py replay descendant
pixi run python .pixi/word-games-v4-review-v1/Neighborhood_Bridge_V4/bridge_v4_audit/run_all_audits.py
```

Run replays/audits on a fresh extracted work copy for a new review. Their exit
status alone is insufficient: inspect full-state counts, freeze checks and the
retained invalid G2 result as recorded here.
