# Broad decoder training improves reconstruction, while task accuracy remains unproven

Training only the decoder on99,367 unique native text frames improved exact
reconstruction on the fixed held-out128 from101 to117 cases. The sixteen fixes
had no exact-reconstruction regressions. All27 cases containing digits now
preserve their digit sequences; the parent preserved26. These are codec
measurements using known target codes, not source-to-summary model accuracy.

| Fixed128 reconstruction metric | Repaired parent | Broadly trained decoder |
| --- | ---: | ---: |
| Exact native strings | 101/128 (78.91%) | 117/128 (91.41%) |
| Matching lexical tokens | 5,264/5,306 (99.21%) | 5,291/5,306 (99.72%) |
| Digit-sequence preservation | 26/27 | 27/27 |
| Empty or invalid-vocabulary outputs | 0 | 0 |

The two complete epochs contain198,734 domain presentations and49,696 author
LM1B retention presentations across6,212 updates. Every domain frame appeared
exactly twice, including the short epoch tails. The independent audit replayed
the domain orders, retention choices and augmentations. The encoder/codebooks
remained frozen, strict checkpoint reload passed, and all128 reference code
hierarchies remained identical to the parent. Independent CPU replay reproduced
the saved strings, per-case metrics, paired fixes and regressions.

The stronger reconstruction gate was not met: it requires at least122 exact
cases and all27 digit sequences. High lexical agreement also conceals meaningful
remaining substitutions, including `Azerbaijani` to `Sicilian`, `flutter` to
`deviation`, and several different replacements of `faeries`. Digit-sequence
preservation alone does not establish correct units, attribution or semantics.

An additional encoding diagnostic compared each of the eleven erroneous active
frames with its correct reference. All eleven pairs have different complete
code vectors. All117 identical-frame controls re-encoded identically. Thus
these particular confusions are distinguishable in the frozen representation;
this test does not prove global injectivity or complete representational
capacity. The serial diagnostic matched all128 original reference encodings.
A batch-of-two comparison changed twelve code IDs in one reference case, so
the collision conclusion uses the original singleton execution shape.

The diagnostic's first attempts remain preserved: v1 assumed a nonexistent
replay producer filename, v2 failed the batched reference-identity check, and
v3 attempted to rewrite an exclusive-create progress file. V4 kept the serial
calculation and wrote separate case diagnostics before committing the complete
record. No failed attempt changed model weights or replaced saved evaluation
outputs.

Separately, the expanded source-supervision arm finished2,048 updates on2,384
pairs, with65,536 presentations and exactly32,768 noisy histories. Its independent
audit replayed the sample order and code corruptions; all2,384 sources were
visited27 or28 times. It has not received its complete semantic DEV evaluation.
The matched source-hierarchy run remains active with its original frozen codec.
The improved codec is not silently installed in that comparison.

The recovered-data context inventory completed all104,503 guarded candidates.
It found context for103,745, including a unique observed role for44,777 of45,378
non-substring targets and58,968 of59,125 substring targets. Date and summarizer
identity metadata are available for44,501 non-substring targets. No conflicting
roles or dates were found among these recovered records. These counts describe
available provenance, not semantic correctness or training admission.

The public epoch4 semantic grader is still active. Its saved attempts show
repeated180-second read timeouts; the gateway liveliness endpoint responded200
in0.017 seconds. This separates basic gateway reachability from completion
transport failures, but does not establish the upstream cause. Invalid judgments
must remain unresolved. The latest completed application semantic DEV result
remains0/128. Final320 is sealed and no inference speedup has been established.

Evidence roots under `data/experiments/ncm_paper_regime`:

- `20261027_recovered_decoder_train_v1`: actual terminal exit0; observer4922,
  chunk`9f98a6`; final checkpoint
  `7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b`.
- `20261027_recovered_decoder_audit_v1`: actual93154/`841151`/0.
- `20261027_recovered_decoder_evaluate_v1`: actual27796/`d05196`/0.
- `20261027_recovered_decoder_replay_v1`: actual76067/`6d9f60`/0.
- `20261028_recovered_decoder_confusion_codes_v4`: actual40028/`a41a42`/0.
- `20261027_task_supervision_expanded_audit_v3`: actual21767/`dfe816`/0.
- `20261027_recovered_pair_context_v1`: original handle4600 recovered from
  this thread's tool log; its actual completion was observed as`f0da8c`/0.
