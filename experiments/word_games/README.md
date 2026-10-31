# WORD_GAMES

A two-agent information firewall experiment using Phi-4 14B. Phi chooses a
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

Run from the repository root with the locked environment:

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
