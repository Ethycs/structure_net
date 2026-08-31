# TinyLLM A5--A7 ordered campaign

**Status:** COMPLETED
**Execution order:** A5, then A6, then A7
**Compute:** CUDA GPU campaigns; no model training

## Frozen outcomes

| Study | Classification | Central result |
| --- | --- | --- |
| A5 | `intrinsic_operator_rank_growth` | The 128-to-256 paired-bit rank cliff is not reproduced by zero padding or exact sector duplication, and fixed alternate bit orders do not materially reduce it. |
| A6 | `shared_and_nested_hierarchy_supported` | Chronological peer interactions use compact shared bases whose stable parent directions are nested in child spaces to numerical precision. |
| A7 | `h2_representation_pass_no_finite_size_compression` | One simultaneous normalization-faithful H2 representation passes at 64--256 tokens, but costs more storage and theoretical work than dense causal attention at 256. |

## Ordered evidence chain

1. A5 preserved A4's strict paired-bit TTNO failure: median rank increased from
   32 at 128 tokens to 90 at 256, with no cell receiving a 25% topology
   reduction.
2. A6 showed this does not rule out a chronological hierarchy: sharing
   inflation and nested augmented-rank ratio were both 1.0 at the gated
   quantiles, with 86.85% stable cuts.
3. A7 converted that diagnostic structure into one actual H2 operator. It
   passed 120/120 cells at lengths 64 and 128 and 118/120 at length 256.
4. A7 failed finite-size compression: median length-256 storage and operation
   ratios were 1.823 and 1.764 relative to dense causal attention.

The resulting target is H2 rather than the tested strict paired-bit TTNO, but a
useful implementation still requires a longer-context crossover and A9-style
implicit core construction.

## Primary artifacts

| Study | Campaign SHA-256 |
| --- | --- |
| A5 | `d9585b4d4d833e632052d2277ecdd11eb2949899aa1073e00d0f006c234f04ef` |
| A6 | `6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01` |
| A7 | `4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2` |

The completed NAL ledger is
`data/hypothesis_registry/tinyllm-dynamic-ttno-followups-v1.json`. All three
hypotheses are persisted as tested with five completed evaluation seeds and no
failed seed.

## A7 sensitivity conclusion

Leaf sizes 8/32, separation ratios 0.5/2, and rank envelopes 0.5x/2x were run
as complete non-rescuing campaigns after the primary checksum was frozen.
Every arm still exceeded dense finite-size cost. Only the half-rank envelope
lost representation success; the double-rank arm was identical to primary.
