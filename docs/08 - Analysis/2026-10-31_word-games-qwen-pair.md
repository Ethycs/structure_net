# WORD_GAMES Qwen2.5 14B / 3B switch

Exploratory result, 2026-10-31. `word-games-qwen25-pair-v1`.
Classification: **`draft_metaphor_gate_failed`**. Model loading and shared-token
speculation work, but this configuration is not ready for the full bridge.
The active research pair is Qwen2.5-14B-Instruct / Qwen2.5-3B-Instruct;
target-only remains the default. No production endpoint was replaced.

Aggregate: `data/experiments/word_games/20261031_qwen_pair_analysis_v1/campaign_results.json`.
SHA256: `f0a2eb745bb5ef1db714ec93c7d7ac7aed16347a2a078ced8f030941b3e13f1b`.
[Prospective protocol](../07%20-%20Status%20Reports/2026-10-31_word-games-qwen-pair-plan.md).

## Model and runtime

Both models are official Qwen Q4_K_M GGUF downloads, used directly without
conversion or training. Exact repository revisions, shard sizes and hashes
are pinned in `experiments/word_games/qwen_pair.py`. The target files total
8.99 GB and the draft is 2.10 GB. The native runtime is llama.cpp b11960 in
the pinned Ubuntu container. Controllers use the locked word-games Pixi
environment from the repository root.

The target runs on RTX 3060 12 GB. The draft runs on RTX 2060 SUPER 8 GB,
as explicitly selected by the user. The existing Qwen3-8B service on the
6 GB RTX 2060 remains running. Loaded GPU totals were 9,508 MiB on the 3060
and 4,882 MiB on the SUPER with speculation, including other resident
allocations. Both owned workers were closed after their experiments.

All 151,936 shared token IDs match. The target's extra 128 entries are unused
padding. Merges, special tokens, pre-tokenizer and chat template match;
333 corpus/control strings produced identical native IDs and byte roundtrips.
The pair needs no token translation map. Evidence:
`20261031_qwen_tokenizers_v1/build/campaign_results.json` under the project
artifact root.

## Measurements

| Configuration | Pilot output tok/s incl. prefill | Pilot decode tok/s | Live joint cases |
| --- | ---: | ---: | --- |
| Qwen14B target only | 28.404 | 32.909 | **5/12 (41.7%)** |
| Qwen14B + Qwen3B, draft length 3 | 32.965 | 39.247 | Stopped at metaphor validation; 0/12 executed |

The three frozen pilot requests show a **16.1% output-rate gain**, with
193/249 proposed tokens accepted (77.5%). Only 1/3 outputs matched the baseline
exactly. Request time was 9.294 s versus 8.342 s; output counts differed
(264 versus 275). This is one three-request pilot per arm, not a full 36-request
speed benchmark or an established speed gain without accuracy loss. Model
loading, hashing, warm-up and artifact I/O are excluded from request time.
The first draft process needed 56.2 s to load and 64.3 s for warm-up on the
older GPU; the binary has no native sm75 kernels and uses PTX compilation.

The target-only live suite completed all twelve independent synthetic cases
with 35 model calls: sender operations **5/12**, boundary outcome **9/12**,
receiver structured state **12/12**, joint **5/12**. No canary or forbidden
source ID appeared in the receiver view/output. The full live call stream
measured 26.946 output tok/s, but it is a different workload from the pilot.
These are structural task checks; an independent semantic judge was not run.

Several target failures were malformed final envelopes, including normal
reserve, prompt injection and changed due date. Other failures proposed wrong
operations for unauthorized requests. The host rejected them. Receiver
accuracy on the resulting blocked state does not rescue sender task failure.

The speculative live run selected `Parking Coordination`, violating the
existing lowercase-only metaphor contract. It stopped before case execution.
That is an end-to-end protocol failure, not a measured 0% case accuracy.
The pilot had shown the same invalid label. No prompt relaxation or retry
was used to obtain a passing result.

## Integrity and reproduction

One initial target startup was stopped before generation because the runtime
strips the template source's final newline. A diagnostic verified the actual
rendered prompt byte for byte. The corrected check permits only that source
normalization and checks exact rendered prompt bytes on every call. Both the
failed startup and source snapshots are retained. Two pilot arms completed;
one live arm completed and the other failed its selector gate.

Run `pixi run --locked -e word-games python -m experiments.word_games.qwen_runtime`
with `--stage pilot` or `--stage live`, a fresh `--output`, and optional
`--draft 3`. The README gives download, tokenizer and paired-fixture commands.
Full source/checkpoint/runtime fingerprints and raw outputs are preserved in
the four roots named by the aggregate. The result stays `confirmed: false`.

The preceding Phi map is retained at
`20261031_phi_token_map_v1/build/campaign_results.json`: 3,570,625 artifact bytes,
byte-exact bidirectional maps, no forward inference validation. Switching to
Qwen superseded its integration; it is not a Phi speed or quality result.
