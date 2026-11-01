# WORD_GAMES normal operation on the deployed Triton model

Exploratory functionality study `word-games-normal-operation-v1`, authorized
by the request to test normal operation. Test live sender retrieval, host
transaction execution, authorized projection and receiver interpretation using
the currently deployed `free_whole1024` model. No cached request replay, draft,
prefix caching, model changes, prompt changes or answer retries.

The model selects its metaphor from the existing public task prompt. Invalid
selection stops the study and counts as a functionality failure. Keep the
existing host disclosure policy, source retrieval restrictions, six sender
turn limit and 1,024-token generation cap. All mutations affect synthetic
local fixture state, never external application data.

Run four existing independent ordinary cases: normal reservation, the alternate
authorized due date, permitted condition annotation, and cancellation of a
seeded reservation. Then run a seven-event workflow on one persistent fixture:
reserve Bronze Heron N1, reserve Bronze Clock N2, extend Heron's date while
keeping N1, annotate Heron, cancel Clock, move Heron to the freed N2, cancel
Heron. Expected operations and ideal resulting state remain host-only. The
model must choose any necessary retrievals and produce each proposal itself.

Grade sender operation fidelity, host decision, actual backend state versus
the oracle, receiver decision/occupied slots, private-canary/source-ID absence,
and invariants separately. Joint structured success requires all applicable
checks, not merely receiver correctness after a rejected sender request.
Count prose fidelity separately with an explicit manual review; no external
semantic judge is assumed. The 11 cases/events are a small synthetic diagnostic,
not a population accuracy estimate.

Continue independent cases after a model-level failure. Stop the dependent
workflow at its first failed event and mark remaining events unexecuted; never
repair actual state with expected answers. Stop the whole run on transport,
malformed/truncated generation, changed source/model/configuration or GPU
headroom below 384 MiB. Record errors and completed counts. Do not alter the
serving configuration or restart services for this study.

Freeze source, model manifest, policy and cases before inference. Save each
request, retrieval, output, proposal, private receipt, authorized packet and
before/after state under a new root. First check deterministic validity of all
seven expected workflow operations and existing firewall/model-boundary tests.
Use Pixi from repository root. Initial evidence:
`data/experiments/word_games/20261101_normal_operation_v1`.

Report measured counts, first failures, unexecuted events, practical supported
scope, and a functional path diagram. Preserve historical evidence and user
worktree changes. Register the conservative outcome with read-back and commit
the durable implementation and report.
