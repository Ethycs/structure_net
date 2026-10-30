# Neighborhood Bridge V2 — indexed synthetic experiment

**Private test prototype, not a production authorization system.** Standard-library
Python; synthetic museum→garden records only, no external effects. V1 at
`/tmp/neighborhood_bridge_20261030` is preserved unchanged.

## Reproduce

```sh
python test_bridge.py                 # original 16 units + 10,000 seeded attempts
python stress_valid.py                # original 10,000 no-reset valid-heavy attempts
python test_v2.py                     # 7 new tests + repeated originals (39 total)
python check_index_workloads.py        # both workloads, independently check indexes
python replay_v1.py /path/to/v1/stress_test  # archived 24 proposals, exact comparison
sha256sum -c BACKEND_SHA256.txt
```

`bridge.py` and `bridge_cli.py` were frozen before new model trials. Initial
and final regression logs are separate. The initial development runs passed;
there are no suppressed initial failures. No subsequent core modifications are
permitted without a new version/freeze and separate trial labeling.

## What changed

- Stored neighborhood membership sets and `(gallery, slot) → occupant` index.
- Stored tokenized ID/title documents used by scoped BM25. Scoring and scoped
  document-frequency computation still scale with the authorized corpus.
- Host-held capability ID sets for exact read/write membership checks.
- Compact interpreter context summary omits the full authorized ID list.
- Bounded record queries sorted by immutable creation sequence, with optional
  neighborhood and creation-event filters.
- Private **experimental** switch for matched neighborhood-write controls.

Mapping, typed operations, effects, rule/approval checks, atomic transactions,
idempotency, stale dependency rejection, and trusted context binding remain V1
semantics. Context summaries are descriptions, not authority: changes to a
returned summary never update stored permissions. Source museum state projects
into garden state; the latter remains non-writable.

## Fixed API / retrieval contract

The trusted host creates contexts with `issue_context(actor=..., scope=[...],
loan_ids=[...], allowed_ops=[...], event_id=..., referents={...})`. This API is
**not** offered to interpreters. Its full return remains backward compatible.
`proposal(context_or_summary, operations, tx_id)` constructs the unchanged
transaction envelope. Host calls `apply(p, expected_context_id=event_context_id)`.

- `context_summary(context_id=...)`: context ID, actor, scope, allowed_ops,
  event_id, referents, revision, rule_versions, `authorized_loan_count`, and
  `authority="host_stored_only"`. **No loan_ids list.**
- `read(ids, context_id=...)`: unchanged exact scoped-ID lookup response with
  revision, rule_versions, records dictionary, missing_ids.
- `search(query, context_id=..., k=5)`: unchanged BM25 candidate response;
  candidates are not an automatic referent binding.
- `occupancy(gallery_id, slot=None, context_id=...)`: revision, gallery_id,
  slots list of `{slot, occupied, loan_id, visibility}`. `visibility` is
  `empty`, `visible`, or `restricted`; hidden occupants have null loan_id but
  occupied=true. Reads still require scoped gallery access in both test arms.
- `query_records(context_id=..., gallery_id=None, order="creation_desc",
  limit=20, after_sequence=None, event_id=None)`: revision, full records list,
  next_after_sequence (null at end), order. Also accepts creation_asc; limit
  1–20. Event filter refers to immutable **creation** event, not last mutation.
  Sequence cursor is stable ordering, **not** a multi-call snapshot; check the
  returned revision for concurrent changes. Retrieval includes creation actor,
  sequence and event provenance. It does not retrieve general audit history.

Read-only CLI: `python bridge_cli.py --session FILE context`, `read ID ...`,
`search WORDS`, `occupancy GAL [SLOT]`, `query '{"limit":2}'`. Sessions remain
trusted-host snapshots; full capabilities live inside the host session file.
An unrestricted child shell can read those files, so this is an API experiment,
**not an OS isolation/security claim**.

Index maintenance removes all changed old entries before adding new ones,
including atomic swaps, transfers, cancel/home return, create and annotate.
Failed operations do not modify indexes. Trusted fixture/snapshot code that
**directly** changes loan/context dictionaries must call `rebuild_indexes()`;
normal `issue_context` and `apply` maintain them automatically. CLI loading
rebuilds indexes. Snapshot format stays V1-compatible and excludes derived indexes.

## Private enforcement switch

Default: `Engine.fixture()` keeps all enforcement ON.
`Engine.fixture(private_test_disable_neighborhood_writes=True)` disables only
write-time **context neighborhood membership** checks. It does NOT disable
actor permission, allowed operation, ID capability, read scope, inherited date,
approval, medium, slot, typed schema, context binding or dependency checks.
This is only for controlled synthetic on/off comparisons, never a production
setting. The switch is not a proposal or CLI option. Unknown gallery validation
still applies. It is intentionally not stored in the V1-compatible source-state
snapshot; matched-test runners must label their arm explicitly.

## Regression findings

- Original 16 unit tests pass; 10,000 seeded mixed attempts: 390 commits and
  9,610 expected rejections, zero assertion failures.
- Original no-reset 10,000 valid-heavy attempts: 8,500 commits and 1,500 slot
  conflicts, final revision and audit length 8,500, zero assertion failures.
- Seven added units cover index transfer/cancel/create/atomic swap, rollback,
  visible/hidden/empty occupancy, scoped provenance pagination, compact summary
  isolation, narrow test-flag effects, and CLI snapshot reload.
- All **24 archived V1 LM proposals** reproduce exact original receipts and
  before/after source snapshots in V2. This is compatibility evidence, not 24
  newly generated model decisions.

New model trials and scaling measurements belong to independently owned
subdirectories; their actual reports, not these regression counts, determine
natural-language accuracy and measured response-size/latency claims. Host
capability storage still grows; retrieval latency includes real scans where
applicable. No blanket constant-time or constant-total-memory claim is made.
