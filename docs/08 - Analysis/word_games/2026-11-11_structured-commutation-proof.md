# WORD_GAMES: bounded structured commutation

Status: completed bounded verification. Study:
`word-games-structured-proof-coding-recovery-v1`.

**All 85,176 declared cases passed, with zero counterexamples.** There were
3,457 committed and 81,719 rejected transactions. Each was replayed once under
the same transaction ID. Complete source state, results, receipts, rejected-state
rollback, encoded state and typed round trips matched. A deliberately corrupted
decoder produced a detected counterexample.

This establishes the tested adapter property. It does not establish a universal
JSON bijection, an independently implemented target Engine, natural-language
equivalence, general coding capability, or information secrecy.

## The extracted table

The table comes from the previously verified synthetic deployed session
`190b56341e34b99b025afd6535e07b03`, read without modifying the service. Its target
is **airline fleet scheduling**. The frozen, executable table is
[verified_reservation_phi.jsonc](../../../experiments/word_games/fixtures/verified_reservation_phi.jsonc).
SHA-256: `4cbfa0679ead783db600f2270c292759d0f13dc4be4f8cd4e0627baf26afe401`.

| Table | Entries | Total on declared source alphabet | Injective |
|---|---:|---|---|
| Field names | 52 | Yes | Yes |
| Enum values | 18 | Yes | Yes |
| Identifier prefixes | 5 | Yes | Yes |

The target prefixes are prefix-free. Forty-four field names are unchanged;
only these eight field names are renamed:

| Source | Target |
|---|---|
| authorized_loan_count | authorized_subject_count |
| galleries | locations |
| gallery_id | location_ref |
| id | record_id |
| loan_id | subject_ref |
| loan_ids | subject_refs |
| loans | subjects |
| to_gallery_id | target_location_ref |

The prefix table is `loan: → aircraft:`, `gal: → route:`,
`rule: → constraint:`, `approval: → gate:`, `dept: → crew:`.
The complete enum table is in the frozen artifact; for example,
`reserve → gate_assignment`, `transfer → route_assignment`,
`cancel → flight_cancellation`, and `metal → narrow_body`.

## Exactly what commutes

Let E be the actual source `Engine.apply`, C the actual `Mapping.encode`, and D
its decoder. Let B be the actual `Bridge.apply`, which retains the source state,
validates target syntax, decodes the target action, supplies the trusted context,
and invokes E. Let π remove the deliberately omitted legacy `target_id` fields.

For each enumerated source state s, context c and action word a, the checker verifies:

```
B(s, C(a), c).source_after = E(s, a, c).source_after
C(B(s, C(a), c).source_after) = C(E(s, a, c).source_after)
D(C(E(s, a, c).source_after)) = π(E(s, a, c).source_after)
```

It also compares returned status/error/revision, full exported source state
including contexts and receipts, rollback, and replay. This is the precise
implementation-level form of the requested commuting diagram. Writing
`Engine(C(x))` literally would be wrong: the source Engine does not accept target
field names, and there is no separately writable target database.

The algebraic reason is direct: on a typed action domain where `D(C(a)) = a`,
the existing adapter executes `E(s, D(C(a)), c) = E(s, a, c)`. Applying C to its
result gives the commuting output. Exhaustive execution checks that the real
validator, field handling, authority, errors, state changes and replay satisfy
those premises throughout the declared finite domain. This is not an independent
verification that E itself implements every desired application requirement.

## Finite domain

- 21 valid initial placements of two fixture loans over no placement or four
  north/archive slots, excluding shared occupied cells; other records remain fixed.
- Eight contexts: full authority, north-only, bronze-only, reserve-only, viewer,
  north-curator, stale revision and stale rule version.
- 22 explicit events covering all five operations, missing/mismatched approvals,
  dates, slots, duplicate creation, denied scope and protected-text annotations.
- Every atomic action word of length 0, 1 or 2: `1 + 22 + 22² = 507` words.
- `21 × 8 × 507 = 85,176` cases, each followed by one identical-ID replay.

The alphabet and exact source hashes are retained in the run's `plan.json`.
The empty word exercises rejection, not a valid no-op. Rejection counts are
coverage of deliberately invalid conditions, not a model failure rate.
Longer transactions, arbitrary new strings/states, forged host contexts, concurrent
HTTP requests, all possible mappings and natural-language instructions are outside
this proof's bounds.

## The unrestricted claim is false

Two actual counterexamples are retained:

1. `{"id":"loan:bronze","target_id":"one"}` and the same object with
   `target_id:"two"` have identical encodings because `target_id` is dropped.
2. The unrestricted strings `"annotate"` and `"maintenance_note"` both encode
   to `"maintenance_note"`: one is translated and the other passes through.

Thus injective symbol tables do not imply an injective transform over arbitrary
JSON. The bounded typed claim passed; the unrestricted claim did not.

## Evidence and reproduction

Artifacts: `data/experiments/word_games/20261111_structured_proof_v1/`.
Case-stream SHA-256:
`7225e2f527734f3d162a5d71d2141a06810a124b2ad02e02b71bc91ed4010071`.
Elapsed CPU verification: 493.42 seconds. Focused tests: 11 passed.
No model inference, training, GPU reservation change or production mutation was
needed for this part. The one model-produced mapping is frozen, not regenerated.

```sh
pixi run -e word-games python -m pytest --confcutdir=tests/word_games -q tests/word_games/test_structured_proof.py
pixi run -e word-games python -m experiments.word_games.structured_proof --output data/experiments/word_games/NEW_STRUCTURED_PROOF
```

Use a new output directory. The separate coding/recovery pilot measures the
limits this structural proof does not cover.
