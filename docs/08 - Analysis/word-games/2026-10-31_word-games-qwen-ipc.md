# Qwen separate-worker IPC reference

Systems validation, 2026-10-31. Classification:
**`two_worker_ipc_reference_validated`**; `confirmed: false`.
Two native llama.cpp workers ran the pinned Qwen14B on RTX 3060 and Qwen3B
on RTX 2060 SUPER. The existing Qwen3-8B Triton service was preserved and its
readiness endpoint returned HTTP 200 after cleanup.

Aggregate: `data/experiments/word_games/20261031_qwen_ipc_v2/campaign_results.json`.
SHA256: `2ede8365fb004a4bbfe69ddf618c8113c5f0a57081e6b44bb0c845094c6e3d70`.
[Prospective plan](../../07%20-%20Status%20Reports/word-games/2026-10-31_word-games-qwen-ipc-plan.md).

Workers used local Unix-domain HTTP sockets in an owner-only directory, with
Docker networking disabled and one explicit GPU per worker. The controller
passed integer token IDs directly. Each draft result was bound to the request,
confirmed-prefix hash and verified pair of tokenizer fingerprints. Model
identity, template, shared IDs and wrong-candidate rejection were checked.

| Warm health round trip, 100 samples | Median | p95 |
| --- | ---: | ---: |
| Target socket | 0.178 ms | 0.205 ms |
| Draft socket | 0.176 ms | 0.204 ms |

These are small control-message latencies. They exclude GPU generation and
do not characterize large payload or concurrency costs. Across all requests,
JSON body traffic was 19,511 bytes sent / 32,526 received for the target and
13,463 / 25,741 for the draft; HTTP headers are excluded.

Three actual draft/target exchanges used 201-, 848- and 303-token prefixes.
The common accepted candidate-prefix lengths were 0, 3 and 0 out of three
tokens. All deliberate first-token mismatches were rejected. These short,
unconstrained reference generations are not a draft acceptance benchmark or
a WORD_GAMES quality evaluation. The first real draft call incurred 48.416 s
of additional cold kernel compilation; later calls took 0.280 and 0.122 s.
A persistent CUDA compilation cache now lives under `.pixi/`, outside data/.

**The target verifier here is an autoregressive correctness reference.** The
native server's internal speculative implementation still expects an in-process
draft model. External-drafter batched verification is not implemented, and no
end-to-end speculative speedup is claimed for the IPC path. The immediate
engineering task after this transport check is a native external-draft hook,
with prefix rollback and target sampler/grammar state preserved.

The first IPC attempt stopped when the target's idle keep-alive socket expired
during draft warm-up. It is retained at `20261031_qwen_ipc_v1`. The client now
reconnects before using a connection idle for over one second; it never silently
retries a generation POST. A local socket regression test covers the failure.
The second attempt completed. All owned workers were stopped afterward.

Reproduce from the repository root:

```sh
pixi run --locked -e word-games python -m experiments.word_games_ipc.run --output data/experiments/word_games/NEW_IPC_RUN
```

The focused WORD_GAMES suite passes 68 tests, including tokenizer contracts,
model identity, stale responses, changed prefixes and idle socket recovery.
Those host tests do not establish model task accuracy.
