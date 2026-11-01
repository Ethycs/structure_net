# WORD_GAMES

A two-agent information firewall experiment. The current research endpoint is
**Empero's Qwen3.8-35B-A3B distillation, Q4_K_M**, served as
`word-games-qwen38-native` at `http://127.0.0.1:8020` through Triton. The
`free_whole1024` configuration uses the 3060 and both 2060 cards, Q8 KV,
Flash Attention, no speculative draft and no public-prefix cache. This
endpoint serves model calls; the host bridge runs through the separate Python
orchestrator. The model chooses a
metaphor from the public task structure before private source details arrive.
The source agent proposes typed actions; a host policy and the imported
Neighborhood Bridge backend validate them. The receiving agent gets only
approved actions and state rendered with the chosen metaphor. Source prose,
raw requests, private notes and backend errors remain on the source side.

The selector's format checks are not a semantic suitability judge. This first
campaign uses one resource-placement task family and twelve independent
synthetic scenarios. The receiver's structured state accuracy, the sender's
operation accuracy, and disclosure checks are reported separately. Permitted
actions and timing remain possible covert channels.

Run ordinary live functionality from the repository root against the already
loaded research endpoint:

```sh
pixi run python -m experiments.word_games_triton.normal_operation --output data/experiments/word_games/NEW_NORMAL_OPERATION
```

This uses actual sender-selected retrievals, host-applied transactions and fresh
receiver calls. It covers four independent ordinary cases followed by seven
dependent events on one persistent synthetic state. It stops the dependent
workflow at its first failed event and never repairs state using expected answers.
Receiver state accuracy, sender fidelity, actual mutation correctness and prose
fidelity are separate measurements.

The [current live functionality tracker](../../docs/08%20-%20Analysis/2026-11-01_word-games-normal-operation.md)
records 4/4 ordinary independent tasks passing, a receiver state omission at
event five of the persistent workflow, and a separate failed name-only lookup.
It distinguishes correct host mutations and firewall rejection from model task
success. Normal operation is not yet reliable across the tested workflow.

The supplied [V3 autonomous mapping bundle has been reviewed](../../docs/08%20-%20Analysis/word-games/2026-11-01_word-games-v3-bundle-review.md).
Its saved 16-event trajectory, 91 mapping probes and four broader controls pass.
Ordinary saved prompt/output pairs fit the current context; fresh mapping creation
needs a higher output cap. Its operational mapping pipeline is not integrated,
and replay success is not a fresh Qwen accuracy result.

The supplied [V4 bundle has been reviewed](../../docs/08%20-%20Analysis/2026-11-01_word-games-v4-bundle-review.md):
its saved trajectories and checkpoint audits replay successfully, but its
generated schema mappings/capsules are not integrated here. It requires adapting
the disclosed view to this project's policy and increasing generation budgets
for fresh mapping/capsule creation. The active vendor remains V2.

## Golf through poker on the central proxy

`golf_poker_proxy.py` runs a small stochastic golf game through a model-generated
poker vocabulary, alongside direct golf controls. The host executes moves and
scores expected-stroke optimality; `audit_golf_poker_proxy.py` independently checks
the trajectory and solves the oracle by policy iteration. This is a separate
task fixture, not deployment of the V3 placement codec.

```sh
pixi run python -m experiments.word_games.golf_poker_proxy --model codex_sdk/gpt-5.6-sol --output data/experiments/word_games/NEW_GOLF_POKER
```

Use a fresh output directory. The central proxy returns zero usage for this SDK
route, so the experiment uses the running local Qwen server's read-only tokenizer
to report visible token-equivalent throughput. It does not measure local GPU
generation speed or native provider tokens. See the
[measured proxy task report](../../docs/08%20-%20Analysis/word-games/2026-11-01_word-games-golf-poker-proxy.md).

## Claude Code research worker behind WORD_GAMES

`claude_research_bridge.py` provides a loopback proxy between a receiving client
and a read-only Claude Code worker, preserving one native conversation ID across
turns. `research_profile.py` defines 36 explicitly authorized repository facts;
the source worker retrieves evidence and the host emits approved values in a
model-chosen archival-ledger schema. Raw worker prose and quotes remain source-side.
This is a new bounded research profile, separate from the placement Engine.

`run_claude_research_thread.py` executes sixteen dependent turns through the proxy;
`audit_claude_research_thread.py` checks the private/public boundary and source
receipts. The [measured thread](../../docs/08%20-%20Analysis/word-games/2026-11-01_word-games-claude-research-thread.md)
released 30/36 unique facts and recalled 11/11 facts without source rereads.
The frozen benchmark executes sixteen queries once. `--resume-from` starts an
interactive continuation in a fresh artifact directory, preserving the native
conversation and mapping. Each completed turn writes a durable checkpoint.
`ask_claude_research_bridge.py` accepts natural-language questions within the
approved catalog; it fetches the next query ID and saves only the authorized
response. The same conversation reached twenty turns across restarts.

```sh
pixi run python -m experiments.word_games.ask_claude_research_bridge --topic all --question 'What does the NCM summarization evidence establish?' --output data/experiments/word_games/NEW_RESEARCH_QUESTION
```

The default URL is `http://127.0.0.1:18091`. Use a fresh output directory.
This remains a bounded fact channel; questions do not authorize arbitrary
worker prose to cross. The report includes restart and reproduction commands.

## Earlier Qwen2.5 target/draft experiment

The earlier pair is Qwen2.5-14B-Instruct / Qwen2.5-3B-Instruct, both Q4_K_M.
Historical reproduction commands use the locked environment:

```sh
pixi run --locked -e word-games test-word-games
pixi run --locked -e word-games python -m experiments.word_games.prepare_qwen_pair --output data/experiments/word_games/NEW_QWEN_DOWNLOAD
pixi run --locked -e word-games python -m experiments.word_games.run_phi_token_map --worker check_qwen_tokenizers --output data/experiments/word_games/NEW_QWEN_TOKENIZERS
pixi run --locked -e word-games python -m experiments.word_games.qwen_runtime --stage pilot --tokenizer-check data/experiments/word_games/NEW_QWEN_TOKENIZERS/build/campaign_results.json --output data/experiments/word_games/NEW_QWEN_PILOT
pixi run --locked -e word-games python -m experiments.word_games.qwen_runtime --stage pilot --draft 3 --output data/experiments/word_games/NEW_QWEN_DRAFT_PILOT
pixi run --locked -e word-games python -m experiments.word_games.qwen_runtime --stage live --output data/experiments/word_games/NEW_QWEN_LIVE
```

The Qwen entry point defaults to target-only until the measured speculative
speed and task-quality gates pass. `--draft 3` selects native draft-simple with
three proposed tokens; the 14B target verifies every proposal. The models use
the RTX 3060 and RTX 2060 SUPER respectively, with unrelated services preserved.
Each run starts and closes its own native server; this does not deploy a
persistent production bridge. The embedded Qwen chat template is applied by the
server. The original Phi transport remains available for historical reproduction.

[Qwen protocol](../../docs/07%20-%20Status%20Reports/2026-10-31_word-games-qwen-pair-plan.md)
defines the new model identity, pinned releases, checks and decision order.

The [measured switch result](../../docs/08%20-%20Analysis/2026-10-31_word-games-qwen-pair.md)
reports 5/12 joint cases for target-only and a metaphor-validation failure with
the draft. The pair is a research configuration, not a production promotion.

`experiments.word_games_ipc.run` provides separate native workers on the 3060
and 2060 SUPER over Unix-domain sockets. It exchanges token IDs and validates
request/prefix identity. Its target verification is an autoregressive reference;
external-drafter batched speculation is not yet implemented. Run it with a fresh
`--output` using the same locked Pixi environment.

## Standalone Bonsai 2 27B

`prepare_bonsai.py` downloads and verifies the pinned official Q2_0 checkpoint.
`bonsai_runtime.py` evaluates it alone on the RTX 3060 with native b11960,
thinking disabled, no draft or lookup, and the existing bridge prompts. It
supports `--stage pilot`, `full` and `live`, each with a fresh `--output`.
Use the locked `word-games` Pixi environment from the repository root.

The [measured Bonsai result](../../docs/08%20-%20Analysis/2026-10-31_word-games-bonsai-naked.md)
is 24.31 output tokens/s over 36 frozen requests and 10/12 joint structured
live cases. It uses less GPU memory than the Qwen target but is slower and adds
unsupported approval claims to receiver summaries. It remains an exploratory
candidate; structured state correctness does not establish semantic fidelity.

## Qwen3.8-35B-A3B distillation

`prepare_qwen38_moe.py` fetches Empero's pinned Q4_K_M GGUF;
`qwen38_moe_runtime.py` tests it with Flash Attention, Q8 K/V cache and exactly
26 CPU expert layers, with remaining computation on the RTX 3060. It supports
`--stage pilot`, `full` and `live` with fresh output directories and the same
locked Pixi environment. Native logs verify the CPU expert placement.

The [measured result](../../docs/08%20-%20Analysis/2026-10-31_word-games-qwen38-moe.md)
is 24.37 full-request output tokens/s and 37.68 decode-only tokens/s. The live
selector reused `dock` for location and deadline, so no live case-accuracy
percentage was measured. A separate frozen-packet diagnostic is explicitly
distinct from end-to-end bridge performance. This is the community distillation,
tested with thinking disabled, not an official Qwen3.8 release evaluation.

Historical Phi workflow:

```sh
pixi run --locked -e word-games test-word-games
pixi run --locked -e word-games word-games import-bundle
pixi run --locked -e word-games word-games regression --output data/experiments/word_games/NEW_REGRESSION
pixi run --locked -e word-games word-games runtime --output data/experiments/word_games/NEW_RUNTIME
pixi run --locked -e word-games word-games run --pilot --identity data/experiments/word_games/NEW_RUNTIME/identity.json --output data/experiments/word_games/NEW_PILOT
pixi run --locked -e word-games word-games run --identity data/experiments/word_games/NEW_RUNTIME/identity.json --output data/experiments/word_games/NEW_FULL
```

Use fresh output directories; original evidence is never overwritten. Import
preserves the supplied archive and its file hashes. Vendored code is unchanged
and checked against `vendor/provenance.json`. Historical model transcripts
are replay controls, not new Phi results. No training occurs.

`runtime` provisions the project-owned `word-games-phi4-14b` Triton endpoint
once, using the cached Q4_K_M checkpoint, Q8 KV and an 8,192-token context.
It exposes only the RTX 3060 to its worker and does not modify existing model
endpoints. To reload or unload this endpoint subsequently, use Triton's
repository model `load` or `unload` API. The host experiment controller uses
this project's Pixi environment; the serving worker uses the existing
container's locked Pixi environment.

[Protocol](../../docs/07%20-%20Status%20Reports/2026-10-30_word-games-phi4-firewall-plan.md)
defines the evidence gates and limitations. Raw artifacts are under
`data/experiments/word_games/`.

## Native Phi-4 speed experiments

`native_speed.py` runs a temporary, GPU-isolated native llama.cpp b11960 server
and closes it after the experiment. It supports the frozen 36-request replay
and the live two-agent campaign. This is an opt-in research path; it does not
replace the existing Triton worker. Check the measured quality gate before
promoting a throughput result.

The CUDA 12.8 binary release is stored at
`.pixi/word-games-native-b11960/llama-b11960`, outside the data store. The
release archive SHA256 is
`64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4`.
The exact measured Ubuntu runtime image is pinned in `native_speed.py` and
available in the local Docker cache. `Dockerfile.native` describes its build;
a rebuilt image needs a new explicit runtime pin and experiment identity.
The cached Phi-4 Q4_K_M checkpoint is hash-checked before every run.

```sh
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.native_speed --stage full --output data/experiments/word_games/NEW_NATIVE_BASELINE
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.native_speed --stage full --spec ngram-simple --ngram 4 --draft 16 --batch 1024 --kv f16 --output data/experiments/word_games/NEW_NATIVE_LOOKUP
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.native_speed --stage live --output data/experiments/word_games/NEW_NATIVE_QUALITY_BASELINE
env CUDA_VISIBLE_DEVICES='' pixi run --locked -e word-games python -m experiments.word_games.native_speed --stage live --spec ngram-simple --ngram 4 --draft 16 --batch 1024 --kv f16 --paired-fixture-root data/experiments/word_games/NEW_NATIVE_QUALITY_BASELINE --output data/experiments/word_games/NEW_NATIVE_QUALITY_LOOKUP
```

Prompt lookup proposes token spans from the current request and its generated
continuation; Phi verifies them. It uses no second model, stored answers or
cross-request prefix cache. Primary throughput counts visible output tokens
over full HTTP request time, including prefill and rejected-draft work. EOS
and unaccepted candidates do not count as output. Native decode-only timing
is a separate secondary metric. Each output directory contains the command,
binary/source hashes, raw responses, server log, memory and aggregate results.

The matched live mode reuses only synthetic fixture context IDs after checking
identical authorization metadata. It never changes production capabilities or
the receiving agent's authorized-only view. The compact-JSON flag preserves a
failed experimental arm for reproducibility; it is not a recommended setting.

[Speed protocol](../../docs/07%20-%20Status%20Reports/2026-10-31_word-games-phi4-40tps-plan.md)
records prospective gates and exploratory amendments.

## MoE placement and CPU tuning

The 2026-10-31 exploratory placement study is implemented in
`moe_placement.py`. It compares CPU thread/affinity settings, eight whole blocks
on RTX 2060 SUPER, an exact-weight expert cache, and routed-expert-only placement
while attention/KV remain on RTX 3060. Use Pixi from the repository root and
fresh output directories. The resident 6 GB RTX 2060 Qwen3-8B is preserved.

Expert-only placement reached 31.02 versus 26.17 full-request tok/s in repeated
36-request replays. Generated answers changed and existing semantic errors
remain; this is not deployment or accuracy approval. See
[the measured report](../../docs/08%20-%20Analysis/2026-10-31_word-games-moe-placement.md)
and its prospective plans. Individual experts split between CPU and GPU within
each layer are a separate, unimplemented research direction.

## Sampling-output acquisition patch

`native_sampling/sampling-output-once.patch` adds a combined sampling-output
view and reuses it during grammar rejection. `prepare_sampling_runtime.py`
builds matched control/candidate libraries from pinned b11960 source;
`check_sampling_output.py` verifies real GPU outputs and synchronization counts;
`run_sampling_output.py` compares the frozen requests with packed38 placement.
Use Pixi from the repository root and fresh output directories. The replay
temporarily unloads Qwen3-8B from the regular 2060 and restores it afterward.

Native tests reduced sampling synchronization from six calls to one, or ten
to one after grammar rejection. All 36 replay outputs matched exactly. Timing
is exploratory: median per-request throughput improved 0.74%, while one slow
control request dominates the larger aggregate difference. See the
[implementation, measurements and reproduction commands](../../docs/08%20-%20Analysis/2026-10-31_word-games-sampling-output-once.md).

## Grammar logits on demand

The next opt-in patch keeps full vocabulary logits on the GPU when an eligible
greedy winner passes grammar. Apply `native_sampling/grammar-on-demand.patch`
and `native_sampling/grammar-partitioned-preview.patch` after the acquisition
patch. `prepare_grammar_runtime.py`, `prepare_grammar_reduction.py`,
`check_grammar_reduction.py` and `run_grammar_reduction.py` build and verify the
candidate with `LLAMA_WORD_GAMES_GREEDY_PREVIEW=1`.

The final two full replays preserved all 72 paired outputs and reduced raw
vocabulary readback by 99.96%. Throughput was 48.08 versus 48.51 tok/s pooled;
the median paired request was about 3% slower. This remains an experimental
transfer optimization, with no production promotion. See the
[measured report and fresh-directory reproduction commands](../../docs/08%20-%20Analysis/2026-10-31_word-games-grammar-on-demand.md).

## Removing the last CPU expert banks

`prepare_expert_compression.py` inventories GGUF tensors and selectively
requantizes banks 24/25. `cpu_expert_elimination.py` compares packed38 against
unchanged-weight all-GPU placement, reduced CPU threads, and the two compressed
banks on the 3060. All other tensor payloads are verified unchanged.

Two full passes per arm measured 49.60 tok/s for packed38, 56.26 for unchanged
all-GPU placement, and 55.65 for selective Q2_K. The unchanged-weight placement
reduced server CPU time per token by 70.5%; compression was unnecessary to fit.
Receiver state checks passed, but existing summary errors remain and live bridge
quality was not established. Qwen3-8B is temporarily unloaded for these tests
and restored afterward. See the
[measured report and reproduction commands](../../docs/08%20-%20Analysis/2026-10-31_word-games-cpu-expert-elimination.md).
