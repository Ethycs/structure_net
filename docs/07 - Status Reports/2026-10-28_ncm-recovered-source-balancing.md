# Source-balanced exposure for the recovered conversational corpus

This CPU-only scheduling qualification prepares the larger paired-data run. It
does not start training or select a model checkpoint, decoder, optimizer, or final
evaluation candidate. Those decisions require a separate fixed training plan and
completed feature/target admission.

The catalog has 38,484 distinct source texts and 43,894 targets. Variant counts
per source are: 33,368 with one, 4,842 with two, 254 with three, and 20 with four.
Use four complete source epochs. Each epoch visits every recovered source once,
choosing the next target in a fixed seeded permutation of that source's variants.
This visits every target at least once while giving every source exactly four
presentations; labels with more historical variants do not multiply source weight.

Retain the entire 2,384-pair reviewed anchor population from the completed
expanded task-supervision catalog. Interleave three recovered-source examples
and one anchor, using complete seeded anchor permutations. Original anchor
features and prompts are retained. Report exact source overlap with the recovered
pool; anchors are intentional additional exposure, not independent new coverage.
This also retains the existing training examples for multi-turn inputs, which
are not represented by the recovered single-message paraphrase pool.

Effective updates contain 32 examples, microbatches contain four, with the final
partial update retained and normalized by its actual size. Seeds are 271142 for
source/anchor ordering, 272142 for target variants, and 273142 for history-noise
flags and streams. Exactly half of each epoch's presentations are designated
noisy. Reuse the existing position-based noise contract; no generated prediction
or teacher response is consulted by the sampler.

Expected exposure is 153,936 recovered presentations and 51,312 reviewed-anchor
presentations across four epochs: 205,248 total, 6,416 updates, and 102,624 noisy
presentations. Recovered targets are noisy historical labels, not factual gold.
This is not a claim that any training update or accuracy improvement occurred.

Freeze source, tests, inputs and the completed source/target bindings into a fresh
root `20261028_recovered_source_balancing_v1`. Emit the entire planned example
stream, independently replay positions and bindings, and report per-source,
per-target and anchor coverage. Verify deterministic resume suffixes, full
variant coverage, partial-update denominators and actual anchor overlap. Locked
Pixi, CPU-only, 4 GB RSS cap, no final-test access or new judge requests. No live
producer, source hierarchy arm, or evaluation history is changed.
