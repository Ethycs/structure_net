# WORD_GAMES

A two-agent information firewall experiment. The active model pair is
**Qwen2.5-14B-Instruct (target) and Qwen2.5-3B-Instruct (draft), Q4_K_M GGUF**.
The target chooses a
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

Run the active Qwen workflow from the repository root with the locked environment:

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
