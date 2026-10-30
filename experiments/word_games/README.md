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
